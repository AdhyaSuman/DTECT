from hashlib import sha256
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Optional
import os

#get_top_words_at_time
from backend.inference.process_beta import get_top_words_at_time

def label_topic_temporal(word_trajectory_str: str, llm, cache_path: Optional[str] = None) -> str:
    """
    Label a dynamic topic by providing the LLM with the top words over time.

    Args:
        word_trajectory_str (str): Formatted keyword evolution string.
        llm: LangChain-compatible LLM instance.
        cache_path (Optional[str]): Path to the cache file (JSON).

    Returns:
        str: Short label for the topic.
    """
    topic_key = sha256(word_trajectory_str.encode()).hexdigest()

    # Load cache
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            label_cache = json.load(f)
    else:
        label_cache = {}

    # Return cached result
    if topic_key in label_cache:
        return label_cache[topic_key]

    # Prompt template
    prompt = ChatPromptTemplate.from_template(
        "You are an expert in topic modeling and temporal data analysis. "
        "Given the top words for a topic across multiple time points, your task is to return a short, specific, descriptive topic label. "
        "Avoid vague, generic, or overly broad labels. Focus on consistent themes in the top words over time. "
        "Use concise noun phrases, 2–5 words max. Do NOT include any explanation, justification, or extra output.\n\n"
        "Top words over time:\n{trajectory}\n\n"
        "Return ONLY the label (no quotes, no extra text):"
    )
    chain = prompt | llm | StrOutputParser()

    try:
        label = chain.invoke({"trajectory": word_trajectory_str}).strip()
    except Exception as e:
        label = "Unknown Topic"
        print(f"[Labeling Error] {e}")

    # Update cache and save
    label_cache[topic_key] = label
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(label_cache, f, indent=2)

    return label


def get_topic_labels(beta, vocab, time_labels, llm, cache_path):
    topic_labels = {}
    for topic_id in range(beta.shape[1]):
        word_trajectory_str = "\n".join([
            f"{time_labels[t]}: {', '.join(get_top_words_at_time(beta, vocab, topic_id, t, top_n=10))}"
            for t in range(beta.shape[0])
        ])
        label = label_topic_temporal(word_trajectory_str, llm=llm, cache_path=cache_path)
        topic_labels[topic_id] = label
    return topic_labels