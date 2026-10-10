"""Aegis Document Intelligence Agent - LLM Client Service.

Provides unified management for Google Gemini and OpenAI models.
Includes explicit API key detection, 401/429/timeout classification,
and prevents infinite/uncontrolled retries on quota or authentication failure.
"""

import os
from typing import Any, Dict, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()


def detect_api_keys() -> Dict[str, str]:
    """Detect available API keys from environment or sidebar input."""
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    return {
        "gemini": gemini_key,
        "openai": openai_key,
        "has_any_key": bool(gemini_key or openai_key),
    }


def classify_llm_error(exc: Exception) -> Tuple[str, str]:
    """Classify model provider exceptions into structured error codes and user-friendly descriptions.
    
    Returns (error_code, description):
    - ('AUTH_ERROR', '...')
    - ('QUOTA_EXHAUSTED', '...')
    - ('RATE_LIMIT', '...')
    - ('NETWORK_TIMEOUT', '...')
    - ('UNKNOWN_ERROR', '...')
    """
    err_str = str(exc).lower()
    err_type = type(exc).__name__.lower()

    if any(k in err_str or k in err_type for k in ["404", "not found", "not_found", "model not found"]):
        return ("MODEL_NOT_FOUND", "The requested model was not found in Gemini API. Please select 'gemini-1.5-flash'.")

    if any(k in err_str or k in err_type for k in ["401", "unauthorized", "invalid_api_key", "invalid api key", "authentication"]):
        return ("AUTH_ERROR", "Invalid API Key or unauthorized access.")

    if any(k in err_str or k in err_type for k in ["429", "quota", "resource_exhausted", "exceeded your current quota", "insufficient_quota"]):
        return ("QUOTA_EXHAUSTED", "API quota or credit limit exhausted for current plan.")

    if any(k in err_str or k in err_type for k in ["rate limit", "ratelimit", "too many requests"]):
        return ("RATE_LIMIT", "API rate limit reached. Please pause requests briefly.")

    if any(k in err_str or k in err_type for k in ["timeout", "timed out", "connection error", "connecterror"]):
        return ("NETWORK_TIMEOUT", "Network request timed out or connection failed.")

    return ("UNKNOWN_ERROR", f"{type(exc).__name__}: {exc}")


def get_llm_model(
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    provider: str = "auto",
    max_retries: int = 1,
) -> Tuple[Optional[Any], str, Optional[str]]:
    """Initialize a LangChain LLM instance for Gemini or OpenAI based on provided parameters.
    
    Returns (llm_instance, provider_used, error_reason).
    """
    keys = detect_api_keys()
    
    key = (api_key or "").strip()
    target_model = (model_name or "").strip()

    # Normalize invalid or placeholder model names
    if target_model in {"gemini-3.8-flash", "gemini-3.5-flash", ""}:
        target_model = "gemini-1.5-flash"

    # Determine provider
    if provider == "openai" or (provider == "auto" and target_model.startswith(("gpt", "o1", "o3"))):
        effective_key = key or keys["openai"]
        if not effective_key:
            return None, "none", "No OpenAI API Key supplied."
        
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=target_model if target_model.startswith(("gpt", "o1", "o3")) else "gpt-4o-mini",
                api_key=effective_key,
                temperature=0.1,
                max_retries=max_retries,
            )
            return llm, "openai", None
        except Exception as exc:
            err_code, desc = classify_llm_error(exc)
            return None, "openai", f"{err_code}: {desc}"

    # Default / Gemini provider
    effective_key = key or keys["gemini"]
    if not effective_key:
        return None, "none", "No Gemini API Key supplied in environment or input."

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        gemini_model = target_model if target_model.startswith("gemini") else "gemini-1.5-flash"
        llm = ChatGoogleGenerativeAI(
            model=gemini_model,
            api_key=effective_key,
            temperature=0.1,
            max_retries=max_retries,
        )
        return llm, "gemini", None
    except Exception as exc:
        err_code, desc = classify_llm_error(exc)
        return None, "gemini", f"{err_code}: {desc}"
