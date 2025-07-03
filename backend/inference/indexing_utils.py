import json
import os
import re
import spacy
from collections import defaultdict

# Load spaCy once
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def has_bigram(tokens, bigram):
    parts = bigram.split('_')
    for i in range(len(tokens) - len(parts) + 1):
        if tokens[i:i + len(parts)] == parts:
            return True
    return False

def build_inverse_lemma_map(docs_file_path, cache_path=None):
    """
    Build or load a mapping from lemma -> set of surface forms seen in corpus.
    If cache_path is provided and exists, loads from it.
    Else builds from scratch and saves to cache_path.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[INFO] Loading cached lemma_to_forms from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
        return {lemma: set(forms) for lemma, forms in raw_map.items()}

    print(f"[INFO] Building inverse lemma map from {docs_file_path}...")
    lemma_to_forms = defaultdict(set)

    with open(docs_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            tokens = tokenize(doc['text'])
            spacy_doc = nlp(" ".join(tokens))
            for token in spacy_doc:
                lemma_to_forms[token.lemma_].add(token.text.lower())

    if cache_path:
        print(f"[INFO] Saving lemma_to_forms to {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({k: list(v) for k, v in lemma_to_forms.items()}, f, indent=2)

    return lemma_to_forms

def build_inverted_index(docs_file_path, vocab_set, lemma_map_path=None):
    vocab_unigrams = {w for w in vocab_set if '_' not in w}
    vocab_bigrams = {w for w in vocab_set if '_' in w}

    # Load or build lemma map
    lemma_to_forms = build_inverse_lemma_map(docs_file_path, cache_path=lemma_map_path)

    index = defaultdict(lambda: defaultdict(list))
    docs = []
    global_seen_words = set()

    with open(docs_file_path, 'r', encoding='utf-8') as f:
        for doc_id, line in enumerate(f):
            doc = json.loads(line)
            text = doc['text']
            timestamp = int(doc['timestamp'])
            docs.append({"text": text, "timestamp": timestamp})

            tokens = tokenize(text)
            token_set = set(tokens)
            seen_words = set()

            # Match all lemma queries using surface forms
            for lemma in vocab_unigrams:
                surface_forms = lemma_to_forms.get(lemma, set())
                if token_set & surface_forms:
                    index[lemma][timestamp].append(doc_id)
                    seen_words.add(lemma)

            for bigram in vocab_bigrams:
                if bigram not in seen_words and has_bigram(tokens, bigram):
                    index[bigram][timestamp].append(doc_id)
                    seen_words.add(bigram)

            global_seen_words.update(seen_words)

            if (doc_id + 1) % 500 == 0:
                missing = vocab_set - global_seen_words
                print(f"[INFO] After {doc_id+1} docs, {len(missing)} vocab words still not seen.")
                print("Example missing words:", list(missing)[:5])

    missing_final = vocab_set - global_seen_words
    if missing_final:
        print(f"[WARNING] {len(missing_final)} vocab words were never found in any document.")
        print("Examples:", list(missing_final)[:10])

    return index, docs, lemma_to_forms

def save_index_to_disk(index, index_path):
    index_clean = {
        word: {str(ts): doc_ids for ts, doc_ids in ts_dict.items()}
        for word, ts_dict in index.items()
    }
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w", encoding='utf-8') as f:
        json.dump(index_clean, f, ensure_ascii=False)

def load_index_from_disk(index_path):
    with open(index_path, 'r', encoding='utf-8') as f:
        raw_index = json.load(f)

    index = defaultdict(lambda: defaultdict(list))
    for word, ts_dict in raw_index.items():
        for ts, doc_ids in ts_dict.items():
            index[word][int(ts)] = doc_ids

    return index

def load_docs(docs_file_path):
    docs = []
    with open(docs_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            docs.append({
                "text": doc["text"],
                "timestamp": int(doc["timestamp"])
            })
    return docs

def load_index(docs_file_path, vocab, index_path=None, lemma_map_path=None):
    if index_path and os.path.exists(index_path):
        index = load_index_from_disk(index_path)
        docs = load_docs(docs_file_path)
        lemma_to_forms = build_inverse_lemma_map(docs_file_path, cache_path=lemma_map_path)
        return index, docs, lemma_to_forms

    index, docs, lemma_to_forms = build_inverted_index(
        docs_file_path,
        set(vocab),
        lemma_map_path=lemma_map_path
    )

    if index_path:
        save_index_to_disk(index, index_path)

    return index, docs, lemma_to_forms
