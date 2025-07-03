import numpy as np
from scipy.special import softmax

def get_interesting_words(beta, vocab, topic_id, top_k_final=10, restrict_to=None):
    """
    Suggests interesting words by prioritizing "bursty" or "emerging" terms, 
    making it effective at capturing important low-probability words.

    This algorithm focuses on the ratio of a word's peak probability to its mean,
    capturing words that show significant growth or have a sudden moment of high
    relevance, even if their average probability is low.
    
    Parameters:
    - beta: np.ndarray (T, K, V) - Topic-word distributions for each timestamp.
    - vocab: list of V words - The vocabulary.
    - topic_id: int - The ID of the topic to analyze.
    - top_k_final: int - The number of words to return.
    - restrict_to: optional list of str - Restricts scoring to a subset of words.
    
    Returns:
    - list of top_k_final interesting words (strings).
    """
    T, K, V = beta.shape
    
    # --- 1. Detect whether softmax is needed ---
    row_sums = beta.sum(axis=2)
    is_prob_dist = np.allclose(row_sums, 1.0, atol=1e-2)

    if not is_prob_dist:
        print("🔁 Beta is not normalized — applying softmax across words per topic.")
        beta = softmax(beta / 1e-3, axis=2)

    # --- 2. Now extract normalized topic slice ---
    topic_beta = beta[:, topic_id, :]        # Shape: (T, V)
    
    # Mean and Peak probability within the topic for each word
    mean_topic = topic_beta.mean(axis=0)     # Shape: (V,)
    peak_topic = topic_beta.max(axis=0)      # Shape: (V,)
    
    # Corpus-wide mean for baseline comparison
    mean_all = beta.mean(axis=(0, 1))        # Shape: (V,)

    # Epsilon to prevent division by zero for words that never appear
    epsilon = 1e-9

    # --- 3. Calculate the three core components of the new score ---
    
    # a) Burstiness Score: How much a word's peak stands out from its own average.
    # This is the key to finding "surprising" words.
    burstiness_score = peak_topic / (mean_topic + epsilon)

    # b) Peak Specificity: How much the word's peak in this topic stands out from
    # its average presence in the entire corpus.
    peak_specificity_score = peak_topic / (mean_all + epsilon)

    # c) Uniqueness Score (same as before): Penalizes words active in many topics.
    active_in_topics = (beta > 1e-5).mean(axis=0)  # Shape: (K, V)
    idf_like = np.log((K + 1) / (active_in_topics.sum(axis=0) + 1)) # Shape: (V,)
    
    # --- 4. Compute Final Interestingness Score ---
    # This score is high for words that are unique, have a high peak relative
    # to their baseline, and whose peak is an unusual event for that word.
    final_scores = burstiness_score * peak_specificity_score * idf_like
    
    # --- 5. Rank and select top words ---
    if restrict_to is not None:
        restrict_set = set(restrict_to)
        word_indices = [i for i, w in enumerate(vocab) if w in restrict_set]
    else:
        word_indices = np.arange(V)

    if not word_indices:
        return []

    # Rank the filtered indices by the final score in descending order
    sorted_indices = sorted(word_indices, key=lambda i: -final_scores[i])
    
    return [vocab[i] for i in sorted_indices[:top_k_final]]


def get_word_trend(beta, vocab, word, topic_id):
    """
    Get the time trend of a word's probability under a specific topic.

    Args:
        beta: np.ndarray of shape (T, K, V)
        vocab: list of vocab words
        word: word to search
        topic_id: index of topic to inspect (0 <= topic_id < K)

    Returns:
        List of word probabilities over time (length T)
    """
    T, K, V = beta.shape
    if word not in vocab:
        raise ValueError(f"Word '{word}' not found in vocab.")
    if not (0 <= topic_id < K):
        raise ValueError(f"Invalid topic_id {topic_id}. Must be between 0 and {K - 1}.")

    word_index = vocab.index(word)
    trend = beta[:, topic_id, word_index]  # shape (T,)
    return trend.tolist()