"""
Dynamic BERTopic for DTECT.

BERTopic (Grootendorst, 2022) learns topics once on the whole corpus
(embeddings -> UMAP -> clustering -> class-based TF-IDF). Its dynamic
variant (`topics_over_time`) keeps the topic assignments fixed and
recomputes class-based TF-IDF (c-TF-IDF) separately for every timestamp.

BERTopic's own `topics_over_time` only returns the top words per topic and
timestamp. DTECT needs a full temporal topic-word tensor beta (T x K x V),
so this trainer repeats the same per-timestamp computation over the full
DTECT vocabulary:

  1. c-TF-IDF of each topic at time t (BERTopic's own `_c_tf_idf(fit=False)`,
     i.e. the fitted vectorizer and the IDF learned on the full corpus)
  2. if global_tuning or evolution_tuning: L1-normalise each topic row
  3. evolution tuning: average row t with the previous timestamp's row
     (topics present at both) -- see note below
  4. global tuning: average with the topic's L1-normalised global c-TF-IDF
     row (the fitted `c_tf_idf_`)
  5. renormalise rows to sum to 1, so beta[t, k] is a distribution over V
     (does not change the word ranking)

Defaults mirror `BERTopic.topics_over_time` in the installed BERTopic (0.17.4):
global_tuning=True, evolution_tuning=True. In that version the evolution-tuning
step assigns into a temporary `.tolil()` copy of the matrix
(`c_tf_idf.tolil()[idx] = ...`) and therefore has no effect. To mirror that
behaviour exactly, evolution tuning is a no-op unless
`evolution_tuning_effective=True`, which applies the averaging the BERTopic
docs describe. `check_against_topics_over_time()` verifies beta's top words
against BERTopic's own output.

A (topic, time) cell with no documents takes the topic's global (all-time)
c-TF-IDF row; the number of such cells is stored in `self.empty_cells_`.

Embeddings: documents are embedded from their raw text (`raw_texts`, aligned
with dataset.train_texts); the preprocessed train_texts are only used for the
c-TF-IDF word counts. Documents longer than the encoder's max_seq_length are
split into token chunks, each chunk is embedded and the chunk embeddings are
mean-pooled per document. Without `raw_texts`, train_texts are embedded.

"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from backend.datasets.utils import _utils
from backend.datasets.utils.logger import Logger

logger = Logger("WARNING")


class BERTopicTrainer:
    def __init__(self,
                 dataset,
                 num_topics=50,
                 num_top_words=15,
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                 clustering="kmeans",          # "kmeans" (fixed K) or "hdbscan" (BERTopic default)
                 min_topic_size=10,            # only used with hdbscan
                 global_tuning=True,          # BERTopic topics_over_time default
                 evolution_tuning=True,       # BERTopic topics_over_time default (no-op, see docstring)
                 evolution_tuning_effective=False,
                 raw_texts=None,              # raw text per train doc (for embeddings)
                 umap_n_neighbors=15,
                 umap_n_components=5,
                 embedding_batch_size=64,
                 seed=42,
                 verbose=False
                ):
        self.dataset = dataset
        self.vocab = list(dataset.vocab)
        self.num_topics = num_topics
        self.num_top_words = num_top_words
        self.embedding_model = embedding_model
        self.clustering = clustering
        self.min_topic_size = min_topic_size
        self.global_tuning = global_tuning
        self.evolution_tuning = evolution_tuning
        self.evolution_tuning_effective = evolution_tuning_effective
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_n_components = umap_n_components
        self.embedding_batch_size = embedding_batch_size
        self.seed = seed
        self.raw_texts = raw_texts
        self.verbose = verbose
        logger.set_level("DEBUG" if verbose else "WARNING")

        self.model = None
        self.beta = None
        self._encoder = None
        self.empty_cells_ = None
        self.chunk_stats_ = None

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def _docs_and_times(self):
        docs = list(self.dataset.train_texts)
        times = self.dataset.train_times
        times = times.cpu().numpy() if hasattr(times, "cpu") else np.asarray(times)
        return docs, times.astype(int)

    def _make_umap(self):
        from umap import UMAP
        return UMAP(n_neighbors=self.umap_n_neighbors,
                    n_components=self.umap_n_components,
                    min_dist=0.0, metric="cosine",
                    random_state=self.seed)

    def _build_model(self, precomputed_reduction=False):
        from bertopic import BERTopic

        if precomputed_reduction:
            # embeddings passed to fit_transform are already UMAP-reduced
            from bertopic.dimensionality import BaseDimensionalityReduction
            umap_model = BaseDimensionalityReduction()
        else:
            umap_model = self._make_umap()

        if self.clustering == "kmeans":
            cluster_model = KMeans(n_clusters=self.num_topics, random_state=self.seed, n_init=10)
        elif self.clustering == "hdbscan":
            from hdbscan import HDBSCAN
            cluster_model = HDBSCAN(min_cluster_size=self.min_topic_size, metric="euclidean",
                                    cluster_selection_method="eom", prediction_data=True)
        else:
            raise ValueError("clustering must be 'kmeans' or 'hdbscan'")

        # Fixed DTECT vocabulary, so beta columns line up with vocab.txt.
        # Texts are already preprocessed and space-separated (bigrams joined with '_').
        vectorizer = CountVectorizer(vocabulary=self.vocab, token_pattern=r"(?u)\S+", lowercase=False)

        # language=None: with language="english" BERTopic strips non-alphanumerics
        # (e.g. '_' in bigrams) before vectorizing, so bigram vocab entries get df=0
        # and an infinite IDF. BERTopic only sets this itself when embedding_model is
        # given, not when precomputed embeddings are passed to train().
        return BERTopic(embedding_model=self._encoder,
                        language=None,
                        umap_model=umap_model,
                        hdbscan_model=cluster_model,
                        vectorizer_model=vectorizer,
                        calculate_probabilities=False,
                        verbose=self.verbose)

    def _load_encoder(self, device=None):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.embedding_model, device=device)
        return self._encoder

    def _chunk(self, text, tokenizer, max_tokens):
        """Split `text` into character spans of <= max_tokens wordpiece tokens,
        cutting only at word starts (not inside '##' continuation pieces)."""
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False, verbose=False)
        offsets, ids = enc["offset_mapping"], enc["input_ids"]
        if len(ids) <= max_tokens:
            return [text]
        pieces = tokenizer.convert_ids_to_tokens(ids)
        chunks, start = [], 0
        while start < len(ids):
            end = min(start + max_tokens, len(ids))
            if end < len(ids):
                back = end
                while back > start + max_tokens // 2 and pieces[back].startswith("##"):
                    back -= 1
                end = back
            chunks.append(text[offsets[start][0]:offsets[end - 1][1]])
            start = end
        return chunks

    def embed(self, texts=None, device=None):
        """Embed documents (default: raw_texts, else train_texts) with chunking + mean pooling.

        Each document is split into chunks of at most (max_seq_length - 2 - 8) wordpiece
        tokens (special tokens + safety margin), all chunks are encoded, and the chunk
        embeddings of a document are averaged (unweighted mean).
        """
        if texts is None:
            texts = self.raw_texts if self.raw_texts is not None else list(self.dataset.train_texts)
        enc = self._load_encoder(device)
        tok = enc.tokenizer
        max_tokens = enc.max_seq_length - 2 - 8

        chunks, owner = [], []
        for d, text in enumerate(texts):
            cs = [c for c in self._chunk(text or "", tok, max_tokens) if c.strip()] or [text or ""]
            chunks.extend(cs)
            owner.extend([d] * len(cs))
        owner = np.asarray(owner)

        chunk_emb = enc.encode(chunks, batch_size=self.embedding_batch_size,
                               show_progress_bar=self.verbose, convert_to_numpy=True)
        n_docs = len(texts)
        counts = np.bincount(owner, minlength=n_docs)
        doc_emb = np.zeros((n_docs, chunk_emb.shape[1]), dtype=np.float64)
        np.add.at(doc_emb, owner, chunk_emb)
        doc_emb = (doc_emb / counts[:, None]).astype(np.float32)

        self.chunk_stats_ = {
            "encoder": self.embedding_model,
            "max_seq_length": int(enc.max_seq_length),
            "chunk_max_tokens": int(max_tokens),
            "n_docs": int(n_docs),
            "n_chunks": int(len(chunks)),
            "n_docs_chunked": int((counts > 1).sum()),
            "chunks_per_doc_mean": float(counts.mean()),
            "chunks_per_doc_max": int(counts.max()),
            "pooling": "mean",
        }
        return doc_emb

    def reduce(self, embeddings):
        """UMAP reduction (seeded); reuse the result across several K via train(reduced_embeddings=...)."""
        return self._make_umap().fit_transform(embeddings)

    def train(self, embeddings=None, reduced_embeddings=None):
        """Fit BERTopic and build beta.

        embeddings:         (N x d) document embeddings; computed with `embed()` if None.
        reduced_embeddings: (N x umap_n_components) precomputed UMAP output; if given,
                            UMAP is skipped and `embeddings` is not needed.
        """
        docs, times = self._docs_and_times()
        if reduced_embeddings is not None:
            self.model = self._build_model(precomputed_reduction=True)
            topics, _ = self.model.fit_transform(docs, embeddings=reduced_embeddings)
        else:
            if embeddings is None:
                embeddings = self.embed()
            self.model = self._build_model()
            topics, _ = self.model.fit_transform(docs, embeddings=embeddings)
        self.topics_ = np.asarray(topics)

        self.beta = self._compute_beta(docs, times, self.topics_)
        return self.get_top_words()

    # ------------------------------------------------------------------ #
    # beta (T x K x V)
    # ------------------------------------------------------------------ #
    def _compute_beta(self, docs, times, topics):
        """beta[t, k] mirroring BERTopic.topics_over_time (see module docstring)."""
        m = self.model
        df = pd.DataFrame({"Document": docs, "Topic": topics, "Time": times})

        # As in topics_over_time: global rows = L1-normalised fitted c_tf_idf_, indexed by the
        # position of the topic among the sorted unique topics (outlier -1 included if present).
        global_all = normalize(m.c_tf_idf_, axis=1, norm="l1", copy=True).toarray()
        all_topics = sorted(df.Topic.unique())
        all_idx = {k: i for i, k in enumerate(all_topics)}

        topic_ids = [k for k in all_topics if k != -1]        # K real topics (-1: HDBSCAN outliers)
        K, V, T = len(topic_ids), len(self.vocab), self.dataset.num_times
        row_of = {k: i for i, k in enumerate(topic_ids)}
        global_rows = global_all[[all_idx[k] for k in topic_ids]]

        beta = np.zeros((T, K, V), dtype=np.float64)
        empty = np.ones((T, K), dtype=bool)
        prev = None                                           # (topic ids, rows) of previous timestamp
        for t in range(T):
            sel = df[(df.Time == t) & (df.Topic != -1)]
            if len(sel):                                      # timestamps without documents are skipped,
                per_topic = (sel.groupby(["Topic"], as_index=False)   # as in topics_over_time
                                .agg({"Document": " ".join}))
                c, _ = m._c_tf_idf(per_topic, fit=False)      # same preprocessing/vectorizer/IDF as BERTopic
                if self.global_tuning or self.evolution_tuning:
                    c = normalize(c, axis=1, norm="l1", copy=False)
                c = c.toarray()
                cur = list(per_topic.Topic.values)

                if self.evolution_tuning and prev is not None and self.evolution_tuning_effective:
                    prev_ids, prev_rows = prev
                    overlap = sorted(set(prev_ids) & set(cur))
                    ci = [cur.index(k) for k in overlap]
                    pi = [prev_ids.index(k) for k in overlap]
                    c[ci] = (c[ci] + prev_rows[pi]) / 2.0
                # else: BERTopic <= 0.17.4 assigns into a temporary .tolil() copy -> no effect

                if self.global_tuning:
                    c = (global_all[[all_idx[k] for k in cur]] + c) / 2.0

                rows = [row_of[k] for k in cur]
                beta[t, rows] = c
                empty[t, rows] = False
                if self.evolution_tuning:
                    prev = (cur, c.copy())

            # (topic, time) cells without documents: the topic's global c-TF-IDF row
            beta[t, empty[t]] = global_rows[empty[t]]

        # Each row becomes a distribution over the vocabulary (without tuning, BERTopic's
        # per-time rows are not L1-normalised; normalising does not change the word ranking)
        sums = beta.sum(-1, keepdims=True)
        sums[sums == 0] = 1.0
        beta = beta / sums
        self.topic_ids = topic_ids
        self.empty_cells_ = {
            "n_empty_cells": int(empty.sum()),
            "n_cells": int(T * K),
            "fraction": float(empty.mean()) if T * K else 0.0,
            "per_time": empty.sum(1).astype(int).tolist(),
            "per_topic": empty.sum(0).astype(int).tolist(),
        }
        return beta.astype(np.float32)

    def check_against_topics_over_time(self, docs=None, times=None, n_words=5):
        """Compare beta's top words with BERTopic's own topics_over_time (same tuning flags).

        Returns the fraction of non-empty (topic, time) cells whose top-`n_words` list is identical
        (ordered), identical as a set, and identical up to ties (the beta values at the two lists'
        positions agree, i.e. differences only come from ordering words with equal scores). Only meaningful for the BERTopic-default flags or
        evolution_tuning_effective=False, because that is what topics_over_time computes.
        """
        if docs is None:
            docs, times = self._docs_and_times()
        tot = self.model.topics_over_time(docs, list(np.asarray(times).astype(int)),
                                          evolution_tuning=self.evolution_tuning,
                                          global_tuning=self.global_tuning)
        row_of = {k: i for i, k in enumerate(self.topic_ids)}
        vidx = {w: i for i, w in enumerate(self.vocab)}
        n = same_order = same_set = up_to_ties = 0
        for r in tot.itertuples():
            if r.Topic not in row_of:
                continue
            ref = [w.strip() for w in r.Words.split(",")][:n_words]
            b = self.beta[int(r.Timestamp), row_of[r.Topic]].astype(np.float64)
            top = np.argsort(-b, kind="stable")[:n_words]
            ours = [self.vocab[i] for i in top]
            n += 1
            same_order += ours == ref
            same_set += set(ours) == set(ref)
            up_to_ties += (ours == ref) or (all(w in vidx for w in ref) and np.allclose(
                np.sort(b[[vidx[w] for w in ref]]), np.sort(b[top]), rtol=1e-5, atol=1e-9))
        return {"n_cells": n, "top_words": n_words,
                "identical_order": same_order / max(n, 1), "identical_set": same_set / max(n, 1),
                "identical_up_to_ties": up_to_ties / max(n, 1)}

    # ------------------------------------------------------------------ #
    # Same interface as the other DTECT trainers
    # ------------------------------------------------------------------ #
    def get_beta(self):
        if self.beta is None:
            raise RuntimeError("Call train() first.")
        return self.beta

    def get_top_words(self, num_top_words=None):
        num_top_words = num_top_words or self.num_top_words
        beta = self.get_beta()
        return [_utils.get_top_words(beta[t], self.vocab, num_top_words, self.verbose)
                for t in range(beta.shape[0])]

    def get_theta(self):
        """One-hot document-topic assignments (BERTopic assigns one topic per document)."""
        theta = np.zeros((len(self.topics_), len(self.topic_ids)), dtype=np.float32)
        row_of = {t: i for i, t in enumerate(self.topic_ids)}
        for d, t in enumerate(self.topics_):
            if t in row_of:
                theta[d, row_of[t]] = 1.0
        return theta

    def save(self, model_dir):
        """Write beta.npy so the DTECT interface can load the model."""
        import os
        os.makedirs(model_dir, exist_ok=True)
        np.save(os.path.join(model_dir, "beta.npy"), self.get_beta())
