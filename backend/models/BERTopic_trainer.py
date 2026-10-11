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

  1. c-TF-IDF of each topic at time t, using the IDF learned on the full corpus
  2. L1-normalise each topic row
  3. evolution tuning (optional, off by default): average row t with row t-1
     (topics present at both t-1 and t)
  4. global tuning: average with the topic's global (all-time) c-TF-IDF
  5. renormalise rows to sum to 1, so beta[t, k] is a distribution over V

Topics with no documents at time t take their row from t-1
(or the global row at t = 0).

evolution_tuning defaults to False: in BERTopic (checked up to 0.17.4) the
evolution-tuning assignment in `topics_over_time` writes into a temporary
`.tolil()` copy and has no effect, so with the default settings beta's top
words match BERTopic's own `topics_over_time(global_tuning=True)`.
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
                 global_tuning=True,
                 evolution_tuning=False,      # see module docstring
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
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_n_components = umap_n_components
        self.embedding_batch_size = embedding_batch_size
        self.seed = seed
        self.verbose = verbose
        logger.set_level("DEBUG" if verbose else "WARNING")

        self.model = None
        self.beta = None

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def _docs_and_times(self):
        docs = list(self.dataset.train_texts)
        times = self.dataset.train_times
        times = times.cpu().numpy() if hasattr(times, "cpu") else np.asarray(times)
        return docs, times.astype(int)

    def _build_model(self):
        from bertopic import BERTopic
        from umap import UMAP

        umap_model = UMAP(n_neighbors=self.umap_n_neighbors,
                          n_components=self.umap_n_components,
                          min_dist=0.0, metric="cosine",
                          random_state=self.seed)

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

    def train(self, embeddings=None):
        """Fit BERTopic and build beta. `embeddings` (N x d) can be passed to skip encoding."""
        docs, times = self._docs_and_times()
        self._encoder = None
        if embeddings is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.embedding_model)
            embeddings = self._encoder.encode(docs, batch_size=self.embedding_batch_size,
                                              show_progress_bar=self.verbose)
        self.model = self._build_model()
        topics, _ = self.model.fit_transform(docs, embeddings=embeddings)
        self.topics_ = np.asarray(topics)

        self.beta = self._compute_beta(docs, times, self.topics_)
        return self.get_top_words()

    # ------------------------------------------------------------------ #
    # beta (T x K x V)
    # ------------------------------------------------------------------ #
    def _ctfidf(self, docs_per_topic):
        """c-TF-IDF of already-joined topic documents, using the fitted IDF."""
        X = self.model.vectorizer_model.transform(docs_per_topic)
        return self.model.ctfidf_model.transform(X)

    def _compute_beta(self, docs, times, topics):
        df = pd.DataFrame({"Document": docs, "Topic": topics, "Time": times})
        df = df[df.Topic != -1]                      # drop HDBSCAN outliers
        topic_ids = sorted(df.Topic.unique())        # K real topics
        K, V, T = len(topic_ids), len(self.vocab), self.dataset.num_times
        row_of = {t: i for i, t in enumerate(topic_ids)}

        # Global c-TF-IDF per topic (all timestamps)
        glob = df.groupby("Topic")["Document"].apply(" ".join)
        global_ctfidf = normalize(self._ctfidf(glob.loc[topic_ids].tolist()), norm="l1", axis=1).toarray()

        beta = np.zeros((T, K, V), dtype=np.float64)
        prev_present = np.zeros(K, dtype=bool)
        for t in range(T):
            sel = df[df.Time == t]
            present = np.zeros(K, dtype=bool)
            if len(sel):
                per_topic = sel.groupby("Topic")["Document"].apply(" ".join)
                rows = [row_of[k] for k in per_topic.index]
                c = normalize(self._ctfidf(per_topic.tolist()), norm="l1", axis=1).toarray()
                beta[t, rows] = c
                present[rows] = True

            # topics with no documents at time t
            fallback = beta[t - 1] if t > 0 else global_ctfidf
            beta[t, ~present] = fallback[~present]

            if self.evolution_tuning and t > 0:
                both = present & prev_present          # as in BERTopic: only topics seen at t-1 and t
                beta[t, both] = (beta[t, both] + beta[t - 1, both]) / 2.0
            if self.global_tuning:
                beta[t, present] = (beta[t, present] + global_ctfidf[present]) / 2.0
            prev_present = present

        # Each row becomes a distribution over the vocabulary
        sums = beta.sum(-1, keepdims=True)
        sums[sums == 0] = 1.0
        beta = beta / sums
        self.topic_ids = topic_ids
        return beta.astype(np.float32)

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
