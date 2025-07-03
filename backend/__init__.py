# === Inference components ===
from .inference.process_beta import (
    load_beta_matrix,
    get_top_words_at_time,
    get_top_words_over_time,
    load_time_labels
    )

from .inference.indexing_utils import load_index
from .inference.word_selector import (
    get_interesting_words,
    get_word_trend
)
from .inference.peak_detector import detect_peaks
from .inference.doc_retriever import (
    load_length_stats,
    get_yearly_counts_for_word,
    get_all_documents_for_word_year,
    deduplicate_docs,
    extract_snippet,
    highlight,
    get_docs_by_ids,
)

# === LLM components ===
from .llm_utils.label_generator import label_topic_temporal, get_topic_labels
from .llm_utils.token_utils import (
    get_token_limit_for_model,
    count_tokens,
    estimate_avg_tokens_per_doc,
    estimate_max_k,
    estimate_max_k_fast
    )
from .llm_utils.summarizer import (
    summarize_docs,
    summarize_multiword_docs,
    ask_multiturn_followup
)
from .llm.llm_router import (
    list_supported_models,
    get_llm
)

# === Dataset utilities ===
from .datasets import dynamic_dataset
from .datasets import preprocess
from .datasets.utils import logger, _utils
from .datasets.data import file_utils, download

# === Evaluation ===
from .evaluation.CoherenceModel_ttc import CoherenceModel_ttc
from .evaluation.eval import TopicQualityAssessor

# === Models ===
from .models.DETM import DETM
from .models.DTM_trainer import DTMTrainer
from .models.CFDTM.CFDTM import CFDTM
from .models.dynamic_trainer import DynamicTrainer

__all__ = [
    # Inference
    "load_beta_matrix", "load_time_labels", "get_top_words_at_time", "get_top_words_over_time",
    "load_index", "get_interesting_words", "get_word_trend", "detect_peaks",
    "load_length_stats", "get_yearly_counts_for_word", "get_all_documents_for_word_year",
    "deduplicate_docs", "extract_snippet", "highlight", "get_docs_by_ids",

    # LLM
    "summarize_docs", "summarize_multiword_docs", "ask_multiturn_followup",
    "get_token_limit_for_model", "list_supported_models", "get_llm",
    "label_topic_temporal", "get_topic_labels", "count_tokens",
    "estimate_avg_tokens_per_doc", "estimate_max_k", "estimate_max_k_fast",

    # Dataset
    "dynamic_dataset", "preprocess", "logger","_utils", "file_utils", "download", 

    # Evaluation
    "CoherenceModel_ttc", "TopicQualityAssessor",

    # Models
    "DETM", "DTMTrainer", "CFDTM", "DynamicTrainer"
]
