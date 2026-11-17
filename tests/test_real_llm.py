"""
Part 7: Tier 2 Test Suite - Real Multimodal LLM Integration
============================================================
File: tests/test_real_llm.py
Concept: Connects to a real live LLM (Google Gemini or OpenAI) using native
Python standard library (urllib.request) without third-party SDK dependencies.

Environment Variables:
    GEMINI_API_KEY: Google Gemini API Key
    OPENAI_API_KEY: OpenAI API Key (fallback)
"""

import json
import os
import sys
import unittest
import urllib.error
import urllib.request
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.loop import AgentLoop
from agent.tools import default_registry


class RealLLMCaller:
    """
    Standard Python HTTP client for calling live LLM APIs.
    Zero external dependencies (uses standard urllib.request).
    """

    def __init__(self, api_key: Optional[str] = None, provider: str = "auto"):
        self.gemini_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.openai_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.provider = provider

    def is_configured(self) -> bool:
        """Returns True if at least one live API key is available."""
        return bool(self.gemini_key or self.openai_key)

    def __call__(self, prompt: str) -> str:
        """Dispatches prompt to live API."""
        if self.gemini_key:
            return self._call_gemini(prompt)
        elif self.openai_key:
            return self._call_openai(prompt)
        else:
            raise RuntimeError("No API key available for RealLLMCaller.")

    def _call_gemini(self, prompt: str) -> str:
        """Calls Google Gemini API (gemini-1.5-flash / gemini-2.0)."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1000},
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candidates = data.get("candidates", [])
                if candidates:
                    return candidates[0]["content"]["parts"][0]["text"]
                return "Final Answer: No response generated."
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise RuntimeError(f"Gemini API error ({e.code}): {error_body}")

    def _call_openai(self, prompt: str) -> str:
        """Calls OpenAI Chat Completion API."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_key}",
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise RuntimeError(f"OpenAI API error ({e.code}): {error_body}")


class TestRealLLMIntegration(unittest.TestCase):
    """Tier 2 live LLM integration test suite."""

    def test_live_llm_if_configured(self):
        caller = RealLLMCaller()
        if not caller.is_configured():
            self.skipTest(
                "Neither GEMINI_API_KEY nor OPENAI_API_KEY is configured in the environment. "
                "Skipping live Tier 2 network test (Tier 1 FakeLLM suite provides full offline coverage)."
            )

        # Run real live agent invocation
        agent = AgentLoop(
            tools=default_registry,
            llm_caller=caller,
            max_steps=5,
            verbose=True,
        )
        response = agent.run("Inspect sample_invoice.png and tell me the document type.")
        self.assertTrue(response.success)
        self.assertGreater(len(response.final_answer), 5)


if __name__ == "__main__":
    unittest.main()

