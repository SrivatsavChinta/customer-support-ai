"""
Ticket classifier backed by a local Ollama model.

Calls Ollama's HTTP API directly — no LangChain, no agent frameworks.
The prompt template lives in prompts/; only the ticket text is injected here.
"""

from __future__ import annotations

import json
import re
from typing import Any

import requests

from config import config
from src.utils import load_prompt, setup_logging

logger = setup_logging()

# Expected shape of a successful classification response
ClassificationResult = dict[str, Any]


def _build_prompt(ticket: str) -> str:
    """Fill the prompt template with the ticket text."""
    template = load_prompt(config.prompt_path)
    return template.replace("{{ticket}}", ticket)


def _call_ollama(prompt: str) -> str:
    """
    Send a generation request to the local Ollama server.

    Uses /api/generate with stream=false so we get one complete response
    instead of managing a token stream at this early stage.
    """
    url = f"{config.ollama_base_url.rstrip('/')}/api/generate"
    payload = {
        "model": config.model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "num_predict": config.max_tokens,
        },
    }

    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")


def _extract_json(text: str) -> ClassificationResult:
    """
    Parse the model output into the expected classification schema.

    Models sometimes wrap JSON in markdown fences or add prose around it.
    We extract the first JSON object rather than requiring a perfect reply.
    """
    # Strip optional markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.replace("```", "").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError(f"Model did not return valid JSON: {text!r}")
        parsed = json.loads(match.group())

    return {
        "category": str(parsed.get("category", "")),
        "confidence": float(parsed.get("confidence", 0.0)),
        "reason": str(parsed.get("reason", "")),
    }


def classify(ticket: str) -> ClassificationResult:
    """
    Classify a single customer support ticket.

    Args:
        ticket: Free-text body of the support ticket.

    Returns:
        Dict with keys: category (str), confidence (float), reason (str).
    """
    if not ticket or not ticket.strip():
        raise ValueError("ticket must be a non-empty string")

    prompt = _build_prompt(ticket.strip())
    logger.debug("Sending classification request to Ollama")
    raw = _call_ollama(prompt)
    result = _extract_json(raw)
    logger.debug("Classification result: %s", result)
    return result
