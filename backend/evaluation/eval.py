# dynamic_topic_quality.py
import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from backend.evaluation.CoherenceModel_ttc import CoherenceModel_ttc
from typing import List, Dict

class TopicQualityAssessor:
    """
    Calculates various quality metrics for dynamic topic models from in-memory data.

    This class provides methods to compute:
    - Temporal Topic Coherence (TTC)
    - Temporal Topic Smoothness (TTS)
    - Temporal Topic Quality (TTQ)
    - Yearly Topic Coherence (TC)
    - Yearly Topic Diversity (TD)
    - Yearly Topic Quality (TQ)
    """

    def __init__(self, topics: List[List[List[str]]], train_texts: List[List[str]], topn: int, coherence_type: str):
        """
        Initializes the TopicQualityAssessor with data in memory.

        Args:
            topics (List[List[List[str]]]): A nested list of topics with structure (T, K, W),
                                           where T is time slices, K is topics, and W is words.
            train_texts (List[List[str]]): A list of tokenized documents for the reference corpus.
            topn (int): Number of top words per topic to consider for calculations.
            coherence_type (str): The type of coherence to calculate (e.g., 'c_npmi', 'c_v').
        """
        # 1. Set texts and dictionary
        self.texts = train_texts
        self.dictionary = Dictionary(self.texts)

        # 2. Process topics
        # User provides topics as (T, K, W) -> List[timestamps][topics][words]
        # Internal representation for temporal evolution is (K, T, W)
        topics_array_T_K_W = np.array(topics, dtype=object)
        if topics_array_T_K_W.ndim != 3:
            raise ValueError(f"Input 'topics' must be a 3-dimensional list/array. Got {topics_array_T_K_W.ndim} dimensions.")
        self.total_topics = topics_array_T_K_W.transpose(1, 0, 2) # Shape: (K, T, W)
        
        # 3. Get dimensions
        self.K, self.T, _ = self.total_topics.shape

        # 4. Create topic groups for smoothness calculation (pairs of topics over time)
        groups = []
        for k in range(self.K):
            time_pairs = []
            for t in range(self.T - 1):
                time_pairs.append([self.total_topics[k, t].tolist(), self.total_topics[k, t+1].tolist()])
            groups.append(time_pairs)
        self.group_topics = np.array(groups, dtype=object)

        # 5. Create yearly topics (T, K, W) for TC/TD calculation
        self.yearly_topics = self.total_topics.transpose(1, 0, 2)
        
        # 6. Set parameters
        self.topn = topn
        self.coherence_type = coherence_type

    def _compute_coherence(self, topics: List[List[str]]) -> List[float]:
        cm = CoherenceModel(
            topics=topics, texts=self.texts, dictionary=self.dictionary,
            coherence=self.coherence_type, topn=self.topn
        )
        return cm.get_coherence_per_topic()

    def _compute_coherence_ttc(self, topics: List[List[str]]) -> List[float]:
        cm = CoherenceModel_ttc(
            topics=topics, texts=self.texts, dictionary=self.dictionary,
            coherence=self.coherence_type, topn=self.topn
        )
        return cm.get_coherence_per_topic()

    def _topic_smoothness(self, topics: List[List[str]]) -> float:
        K = len(topics)
        if K <= 1:
            return 1.0 # Or 0.0, depending on definition. A single topic has no other topic to be dissimilar to.
        scores = []
        for i, base in enumerate(topics):
            base_set = set(base[:self.topn])
            others = [other for j, other in enumerate(topics) if j != i]
            if not others:
                return 1.0
            overlaps = [len(base_set & set(other[:self.topn])) / self.topn for other in others]
            scores.append(sum(overlaps) / len(overlaps))
        return float(sum(scores) / K)

    def get_ttq_dataframe(self) -> pd.DataFrame:
        """Computes and returns a DataFrame with detailed TTQ metrics per topic chain."""
        all_coh_scores, avg_coh_scores = [], []
        for k in range(self.K):
            coh_per_topic = self._compute_coherence_ttc(self.total_topics[k].tolist())
            all_coh_scores.append(coh_per_topic)
            avg_coh_scores.append(float(np.mean(coh_per_topic)))

        all_smooth_scores, avg_smooth_scores = [], []
        for k in range(self.K):
            pair_scores = [self._topic_smoothness(pair) for pair in self.group_topics[k]]
            all_smooth_scores.append(pair_scores)
            avg_smooth_scores.append(float(np.mean(pair_scores)))
            
        df = pd.DataFrame({
            'topic_idx': list(range(self.K)),
            'temporal_coherence': all_coh_scores,
            'temporal_smoothness': all_smooth_scores,
            'avg_temporal_coherence': avg_coh_scores,
            'avg_temporal_smoothness': avg_smooth_scores
        })
        df['ttq_product'] = df['avg_temporal_coherence'] * df['avg_temporal_smoothness']
        return df

    def get_tq_dataframe(self) -> pd.DataFrame:
        """Computes and returns a DataFrame with detailed TQ metrics per time slice."""
        all_coh, avg_coh, div = [], [], []
        for t in range(self.T):
            yearly_t_topics = self.yearly_topics[t].tolist()
            coh_per_topic = self._compute_coherence(yearly_t_topics)
            all_coh.append(coh_per_topic)
            avg_coh.append(float(np.mean(coh_per_topic)))
            div.append(1 - self._topic_smoothness(yearly_t_topics))
            
        df = pd.DataFrame({
            'year': list(range(self.T)),
            'all_coherence': all_coh,
            'avg_coherence': avg_coh,
            'diversity': div
        })
        df['tq_product'] = df['avg_coherence'] * df['diversity']
        return df

    def get_ttc_score(self) -> float:
        """Calculates the overall Temporal Topic Coherence (TTC)."""
        ttq_df = self.get_ttq_dataframe()
        return ttq_df['avg_temporal_coherence'].mean()

    def get_tts_score(self) -> float:
        """Calculates the overall Temporal Topic Smoothness (TTS)."""
        ttq_df = self.get_ttq_dataframe()
        return ttq_df['avg_temporal_smoothness'].mean()

    def get_ttq_score(self) -> float:
        """Calculates the overall Temporal Topic Quality (TTQ)."""
        ttq_df = self.get_ttq_dataframe()
        return ttq_df['ttq_product'].mean()

    def get_tc_score(self) -> float:
        """Calculates the overall yearly Topic Coherence (TC)."""
        tq_df = self.get_tq_dataframe()
        return tq_df['avg_coherence'].mean()

    def get_td_score(self) -> float:
        """Calculates the overall yearly Topic Diversity (TD)."""
        tq_df = self.get_tq_dataframe()
        return tq_df['diversity'].mean()

    def get_tq_score(self) -> float:
        """Calculates the overall yearly Topic Quality (TQ)."""
        tq_df = self.get_tq_dataframe()
        return tq_df['tq_product'].mean()

    def get_dtq_summary(self) -> Dict[str, float]:
        """
        Computes all dynamic topic quality metrics and returns them in a dictionary.
        """
        ttq_df = self.get_ttq_dataframe()
        tq_df = self.get_tq_dataframe()
        summary = {
            'TTC': ttq_df['avg_temporal_coherence'].mean(),
            'TTS': ttq_df['avg_temporal_smoothness'].mean(),
            'TTQ': ttq_df['ttq_product'].mean(),
            'TC': tq_df['avg_coherence'].mean(),
            'TD': tq_df['diversity'].mean(),
            'TQ': tq_df['tq_product'].mean()
        }
        return summary