"""Swappable language model layer.

* rule    deterministic parser and template writer. Default, used in tests, CI and for every
          number in the README.
* ollama  any local model served by Ollama (for example qwen2.5:1.5b-instruct on a laptop CPU).
* local   a small open model run in process with transformers on CPU (default
          Qwen/Qwen2.5-0.5B-Instruct, about 1 GB, needs the [local-llm] extra).
* openai  any OpenAI compatible endpoint (vLLM, llama.cpp server, a hosted API). Needs
          UPW_LLM_BASE_URL and, if the server wants one, UPW_LLM_API_KEY.

Model output is never trusted directly: parse results are validated with Pydantic and written
answers go through the grounding check in agent.py, with the rule writer as the fallback.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger(__name__)

Intent = Literal["impact", "stages", "travel", "downstream", "upstream", "other"]
STAGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:ft|feet|foot)\b", re.I)


class ParsedQuestion(BaseModel):
    intent: Intent = "other"
    stage_ft: float | None = Field(default=None, ge=-20, le=100)


def rule_parse(question: str) -> ParsedQuestion:
    q = question.lower()
    match = STAGE_RE.search(question)
    stage = float(match.group(1)) if match else None
    if any(w in q for w in ("how long", "travel time", "how many hours", "take to reach", "get from")):
        intent: Intent = "travel"
    elif any(w in q for w in ("upstream", "watch for", "early warning", "feeds into", "flow into")):
        intent = "upstream"
    elif any(w in q for w in ("downstream", "next gauge", "where does", "which gauges will", "expect it next")):
        intent = "downstream"
    elif any(w in q for w in ("flood stage", "flood category", "in flood", "minor flood", "action stage")):
        intent = "stages" if stage is None else "impact"
    elif stage is not None or any(w in q for w in ("what happens", "impact", "flooded")):
        intent = "impact"
    else:
        intent = "other"
    return ParsedQuestion(intent=intent, stage_ft=stage)


class Provider(Protocol):
    name: str

    def parse(self, question: str) -> ParsedQuestion: ...

    def write(self, question: str, context: str, draft: str) -> str: ...


class RuleProvider:
    name = "rule"

    def parse(self, question: str) -> ParsedQuestion:
        return rule_parse(question)

    def write(self, question: str, context: str, draft: str) -> str:
        return draft


PARSE_PROMPT = """Classify the question about river gauges. Reply with JSON only:
{{"intent": one of ["impact","stages","travel","downstream","upstream","other"], "stage_ft": number or null}}
impact = what happens at a gauge at a stage; stages = flood stage thresholds; travel = crest travel time
between two gauges; downstream = which gauges are next downstream; upstream = which gauges feed a gauge.
Question: {question}"""

WRITE_PROMPT = """You write short operational answers for a river forecast desk.
Use only the facts in CONTEXT. Keep every number exactly as written there. Cite document ids in
square brackets. If the context is not enough, say so. Three sentences at most.

CONTEXT:
{context}

DRAFT (correct but terse, rewrite it for a duty officer):
{draft}

QUESTION: {question}
ANSWER:"""


class _HttpProvider:
    name = "http"

    def __init__(self, model: str, base_url: str, timeout_s: float = 120.0) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(timeout=timeout_s)

    def _complete(self, prompt: str) -> str:  # pragma: no cover - needs a running server
        raise NotImplementedError

    def parse(self, question: str) -> ParsedQuestion:
        try:
            raw = self._complete(PARSE_PROMPT.format(question=question))
            blob = re.search(r"\{.*\}", raw, re.S)
            parsed = ParsedQuestion.model_validate(json.loads(blob.group(0) if blob else raw))
        except (httpx.HTTPError, ValueError, ValidationError) as exc:
            log.warning("%s parse failed, using rules: %s", self.name, exc)
            return rule_parse(question)
        # The stage number is read from the text either way, models sometimes drop it.
        if parsed.stage_ft is None:
            parsed.stage_ft = rule_parse(question).stage_ft
        return parsed

    def write(self, question: str, context: str, draft: str) -> str:
        try:
            return self._complete(WRITE_PROMPT.format(context=context, draft=draft, question=question)).strip()
        except httpx.HTTPError as exc:
            log.warning("%s write failed, keeping the draft: %s", self.name, exc)
            return draft


class OllamaProvider(_HttpProvider):
    name = "ollama"

    def _complete(self, prompt: str) -> str:  # pragma: no cover - needs a running server
        resp = self._http.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
        )
        resp.raise_for_status()
        return resp.json()["response"]


class OpenAICompatProvider(_HttpProvider):
    name = "openai"

    def _complete(self, prompt: str) -> str:  # pragma: no cover - needs a running server
        headers = {}
        key = os.environ.get("UPW_LLM_API_KEY", "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        resp = self._http.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={"model": self.model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class LocalHFProvider(_HttpProvider):
    """Runs a small instruct model in process. Greedy decoding so reruns are repeatable."""

    name = "local"

    def __init__(self, model: str = "Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens: int = 160) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = model
        self.max_new_tokens = max_new_tokens
        self._tok = AutoTokenizer.from_pretrained(model)
        self._lm = AutoModelForCausalLM.from_pretrained(model)
        self._lm.eval()

    def _complete(self, prompt: str) -> str:
        import torch

        text = self._tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = self._tok(text, return_tensors="pt")
        with torch.no_grad():
            out = self._lm.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self._tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    def parse(self, question: str) -> ParsedQuestion:
        try:
            return super().parse(question)
        except Exception as exc:  # noqa: BLE001 - any model failure falls back to the rules
            log.warning("local parse failed, using rules: %s", exc)
            return rule_parse(question)


def make_provider(name: str, model: str = "", base_url: str = "") -> Provider:
    if name == "local":
        return LocalHFProvider(model if "/" in model else "Qwen/Qwen2.5-0.5B-Instruct")
    if name == "ollama":
        return OllamaProvider(model, base_url or "http://localhost:11434")
    if name == "openai":
        return OpenAICompatProvider(model, base_url)
    return RuleProvider()
