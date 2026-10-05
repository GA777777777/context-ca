"""LLM client with token accounting.

Supports two backends, chosen from environment variables (see .env.example):
  * Azure OpenAI:  AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY + AZURE_OPENAI_DEPLOYMENT
  * OpenAI-compatible gateway: OPENAI_BASE_URL + OPENAI_API_KEY + LLM_MODEL

Every call is appended to a JSONL usage log with prompt/completion token counts,
because the ProGraph harness discards usage and the pilot needs cost numbers.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"))
except Exception:
    pass


class LLM:
    def __init__(self, usage_log: Optional[str | Path] = None, tag: str = ""):
        from openai import AzureOpenAI, OpenAI

        self.tag = tag
        self._lock = threading.Lock()
        self.n_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.usage_log = Path(usage_log) if usage_log else None
        if self.usage_log:
            self.usage_log.parent.mkdir(parents=True, exist_ok=True)

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if endpoint:
            self.client = AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=endpoint,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
            )
            self.model = os.environ["AZURE_OPENAI_DEPLOYMENT"]
            self.backend = "azure"
        else:
            self.client = OpenAI(
                base_url=os.getenv("OPENAI_BASE_URL") or None,
                api_key=os.environ["OPENAI_API_KEY"],
            )
            self.model = os.environ["LLM_MODEL"]
            self.backend = "openai-compatible"
        # gpt-5.x deployments reject temperature; only send it when explicitly configured.
        t = os.getenv("LLM_TEMPERATURE")
        self.temperature = float(t) if t not in (None, "") else None
        self.max_tokens = int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "2048"))
        # gpt-5.x reasoning models: cap hidden reasoning tokens (minimal|low|medium|high)
        self.reasoning_effort = os.getenv("LLM_REASONING_EFFORT") or None

    def __call__(self, system_prompt: str, user_prompt: str, temperature: float | None = None, tag: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
        if not messages:
            return ""
        kwargs = dict(model=self.model, messages=messages, max_completion_tokens=self.max_tokens)
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        last = None
        for attempt in range(8):
            try:
                t0 = time.time()
                resp = self.client.chat.completions.create(**kwargs)
                dt = time.time() - t0
                text = resp.choices[0].message.content or ""
                u = getattr(resp, "usage", None)
                pt = getattr(u, "prompt_tokens", 0) or 0
                ct = getattr(u, "completion_tokens", 0) or 0
                with self._lock:
                    self.n_calls += 1
                    self.prompt_tokens += pt
                    self.completion_tokens += ct
                    if self.usage_log:
                        with open(self.usage_log, "a", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "ts": time.time(), "tag": tag or self.tag, "model": self.model,
                                "prompt_tokens": pt, "completion_tokens": ct, "latency_s": round(dt, 3),
                            }) + "\n")
                return text
            except Exception as e:  # noqa: BLE001
                last = e
                msg = str(e)
                if "temperature" in msg and "temperature" in kwargs:
                    kwargs.pop("temperature")
                    continue
                if "max_completion_tokens" in msg and "max_completion_tokens" in kwargs:
                    kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                    continue
                if "reasoning_effort" in msg and "reasoning_effort" in kwargs:
                    kwargs.pop("reasoning_effort")
                    continue
                m = re.search(r"try again in ([0-9.]+)\s*s", msg)
                if m or "429" in msg or "rate_limit" in msg:
                    wait = float(m.group(1)) + 1.0 if m else 5.0 * (attempt + 1)
                    time.sleep(min(wait, 60.0))
                    continue
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after retries: {last}")

    def stats(self) -> dict:
        return {
            "backend": self.backend, "model": self.model, "calls": self.n_calls,
            "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
        }
