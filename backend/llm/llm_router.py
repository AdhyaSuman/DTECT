from langchain_anthropic import ChatAnthropic
from backend.llm.custom_mistral import ChatMistral
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import os
import google.auth.transport.requests
import requests


def list_supported_models(provider=None):
    if provider == "OpenAI":
        return ["gpt-4.1-nano", "gpt-4o-mini"]
    elif provider == "Anthropic":
        return ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]
    elif provider == "Gemini":
        return ["gemini-2.0-flash-lite", "gemini-1.5-flash"]
    elif provider == "Mistral":
        return ["mistral-small", "mistral-medium"]
    else:
        # Default fallback: all models grouped by provider
        return {
            "OpenAI": ["gpt-4.1-nano", "gpt-4o-mini"],
            "Anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229"],
            "Gemini": ["gemini-2.0-flash-lite", "gemini-1.5-flash"],
            "Mistral": ["mistral-small", "mistral-medium"]
        }


def get_llm(provider: str, model: str, api_key: str = None):
    if provider == "OpenAI":
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OpenAI API key.")
        return ChatOpenAI(model_name=model, temperature=0, openai_api_key=api_key)

    elif provider == "Anthropic":
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Missing Anthropic API key.")
        return ChatAnthropic(model=model, temperature=0, anthropic_api_key=api_key)

    elif provider == "Gemini":
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing Gemini API key.")
        # --- Patch: Set proxy if available ---
        if "HTTP_PROXY" in os.environ or "http_proxy" in os.environ:

            proxies = {
                "http": os.getenv("http_proxy") or os.getenv("HTTP_PROXY"),
                "https": os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")
            }

            google.auth.transport.requests.requests.Request = lambda *args, **kwargs: requests.Request(
                *args, **kwargs, proxies=proxies
            )

        return ChatGoogleGenerativeAI(model=model, temperature=0, google_api_key=api_key)


    elif provider == "Mistral":
        api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Missing Mistral API key.")
        return ChatMistral(model=model, temperature=0, mistral_api_key=api_key)

    else:
        raise ValueError(f"Unsupported provider: {provider}")

