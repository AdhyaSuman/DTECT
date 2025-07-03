import numpy as np
from tqdm import tqdm
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from backend.datasets.data.file_utils import split_text_word
from typing import List


def coherence(
        reference_corpus: List[str],
        vocab: List[str],
        top_words: List[str],
        coherence_type='c_npmi',
        topn=20
    ):
    flatten_topics = [item for sublist in top_words for item in sublist]
    split_top_words = split_text_word(flatten_topics)
    split_reference_corpus = split_text_word(reference_corpus)
    dictionary = Dictionary(split_text_word(vocab))

    cm = CoherenceModel(
        texts=split_reference_corpus,
        dictionary=dictionary,
        topics=split_top_words,
        topn=topn,
        coherence=coherence_type,
    )
    cv_per_topic = cm.get_coherence_per_topic()
    score = np.mean(cv_per_topic)

    return score