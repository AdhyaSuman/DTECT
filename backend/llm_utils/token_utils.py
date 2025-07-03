from typing import Literal
import tiktoken
import anthropic
from typing import List

# Gemini requires the Vertex AI SDK
try:
    from vertexai.preview import tokenization as vertex_tokenization
except ImportError:
    vertex_tokenization = None

# Mistral requires the SentencePiece tokenizer
try:
    import sentencepiece as spm
except ImportError:
    spm = None

# ---------------------------
# Individual Token Counters
# ---------------------------

def count_tokens_openai(text: str, model_name: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")  # fallback
    return len(encoding.encode(text))

def count_tokens_anthropic(text: str, model_name: str) -> int:
    try:
        client = anthropic.Anthropic()
        response = client.messages.count_tokens(
            model=model_name,
            messages=[{"role": "user", "content": text}]
        )
        return response['input_tokens']
    except Exception as e:
        raise RuntimeError(f"Anthropic token counting failed: {e}")

def count_tokens_gemini(text: str, model_name: str) -> int:
    if vertex_tokenization is None:
        raise ImportError("Please install vertexai: pip install google-cloud-aiplatform[tokenization]")
    try:
        tokenizer = vertex_tokenization.get_tokenizer_for_model("gemini-1.5-flash-002")
        result = tokenizer.count_tokens(text)
        return result.total_tokens
    except Exception as e:
        raise RuntimeError(f"Gemini token counting failed: {e}")

def count_tokens_mistral(text: str) -> int:
    if spm is None:
        raise ImportError("Please install sentencepiece: pip install sentencepiece")
    try:
        sp = spm.SentencePieceProcessor()
        # IMPORTANT: You must provide the correct path to the tokenizer model file
        sp.load("mistral_tokenizer.model")
        tokens = sp.encode(text, out_type=str)
        return len(tokens)
    except Exception as e:
        raise RuntimeError(f"Mistral token counting failed: {e}")

# ---------------------------
# Unified Token Counter
# ---------------------------

def count_tokens(text: str, model_name: str, provider: Literal["OpenAI", "Anthropic", "Gemini", "Mistral"]) -> int:
    if provider == "OpenAI":
        return count_tokens_openai(text, model_name)
    elif provider == "Anthropic":
        return count_tokens_anthropic(text, model_name)
    elif provider == "Gemini":
        return count_tokens_gemini(text, model_name)
    elif provider == "Mistral":
        return count_tokens_mistral(text)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def get_token_limit_for_model(model_name, provider):
    # Example values; update as needed for your providers
    if provider == "openai":
        if "gpt-4.1-nano" in model_name:
            return 1047576  # Based on search results
        elif "gpt-4o-mini" in model_name:
            return 128000 # Based on search results
    elif provider == "anthropic":
        if "claude-3-opus" in model_name:
            return 200000 # Based on search results
        elif "claude-3-sonnet" in model_name:
            return 200000 # Based on search results
    elif provider == "gemini":
        if "gemini-2.0-flash-lite" in model_name:
            return 1048576 # Based on search results
        elif "gemini-1.5-flash" in model_name:
            return 1048576 # Based on search results
    elif provider == "mistral":
        if "mistral-small" in model_name:
            return 32000 # Based on search results
        elif "mistral-medium" in model_name:
            return 32000 # Based on search results
    return 8000  # default fallback


def estimate_avg_tokens_per_doc(
    docs: List[str],
    model_name: str,
    provider: Literal["OpenAI", "Anthropic", "Gemini", "Mistral"]
) -> float:
    """
    Estimate the average number of tokens per document for the given model.

    Args:
        docs (List[str]): List of documents.
        model_name (str): Model name.
        provider (Literal): LLM provider.

    Returns:
        float: Average number of tokens per document.
    """
    if not docs:
        return 0.0
    token_counts = [count_tokens(doc, model_name, provider) for doc in docs]
    return sum(token_counts) / len(token_counts)

def estimate_max_k(
    docs: List[str],
    model_name: str,
    provider: Literal["OpenAI", "Anthropic", "Gemini", "Mistral"],
    margin_ratio: float = 0.1,
) -> int:
    """
    Estimate the maximum number of documents that can fit in the context window.

    Returns:
        int: Estimated K.
    """
    if not docs:
        return 0

    max_tokens = get_token_limit_for_model(model_name, provider)
    margin = int(max_tokens * margin_ratio)
    available_tokens = max_tokens - margin

    avg_tokens_per_doc = estimate_avg_tokens_per_doc(docs, model_name, provider)
    if avg_tokens_per_doc == 0:
        return 0

    return min(len(docs), int(available_tokens // avg_tokens_per_doc))

def estimate_max_k_fast(docs, margin_ratio=0.1, max_tokens=8000, model_name="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model_name)
    avg_len = sum(len(enc.encode(doc)) for doc in docs[:20]) / min(20, len(docs))
    margin = int(max_tokens * margin_ratio)
    available = max_tokens - margin
    return min(len(docs), int(available // avg_len))

def estimate_k_max_from_word_stats(
    avg_words_per_doc: float,
    margin_ratio: float = 0.1,
    avg_tokens_per_word: float = 1.3,
    model_name=None,
    provider=None
) -> int:
    model_token_limit = get_token_limit_for_model(model_name, provider)
    effective_limit = int(model_token_limit * (1 - margin_ratio))
    est_tokens_per_doc = avg_words_per_doc * avg_tokens_per_word
    return int(effective_limit // est_tokens_per_doc)