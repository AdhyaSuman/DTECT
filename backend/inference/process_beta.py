import numpy as np
import json

def load_beta_matrix(beta_path: str, vocab_path: str):
    """
    Loads the beta matrix (T x K x V) and vocab list.

    Returns:
        beta: np.ndarray of shape (T, K, V)
        vocab: list of words
    """
    beta = np.load(beta_path)  # shape: T x K x V
    with open(vocab_path, 'r') as f:
        vocab = [line.strip() for line in f.readlines()]
    return beta, vocab

def get_top_words_at_time(beta, vocab, topic_id, time, top_n):
    topic_beta = beta[time, topic_id, :]
    top_indices = topic_beta.argsort()[-top_n:][::-1]
    return [vocab[i] for i in top_indices]

def get_top_words_over_time(beta, vocab, topic_id, top_n):
    topic_beta = beta[:, topic_id, :]
    mean_beta = topic_beta.mean(axis=0)
    top_indices = mean_beta.argsort()[-top_n:][::-1]
    return [vocab[i] for i in top_indices]

def load_time_labels(time2id_path):
    with open(time2id_path, 'r') as f:
        time2id = json.load(f)
    # Invert and sort by id
    id2time = {v: k for k, v in time2id.items()}
    return [id2time[i] for i in sorted(id2time)]