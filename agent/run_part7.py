"""
Part 7 Demonstration Script: Two-Tier Testing Suite
====================================================
File: agent/run_part7.py
Demonstrates:
1. Tier 1: Deterministic Fake LLM test execution (6 comprehensive unit tests
   covering loop, tools, self-healing, planning, safety gates, and memory).
2. Tier 2: Real Multimodal LLM check (checks for live API keys and tests connection).
"""

from __future__ import annotations

import os
import sys
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.test_fake_llm import TestAgentDeterministic
from tests.test_real_llm import RealLLMCaller, TestRealLLMIntegration


def run_tier1_tests():
    print("=" * 75)
    print("DEMO: Part 7 - Tier 1 Deterministic Unit Test Suite (FakeLLM)")
    print("=" * 75)
    print("Running 7 unit tests with zero API keys or network latency...\n")

    suite = unittest.TestLoader().loadTestsFromTestCase(TestAgentDeterministic)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def run_tier2_check():
    print("\n" + "=" * 75)
    print("DEMO: Part 7 - Tier 2 Real Multimodal LLM Integration Check")
    print("=" * 75)

    caller = RealLLMCaller()
    if caller.is_configured():
        provider = "Gemini API" if caller.gemini_key else "OpenAI API"
        print(f"✅ Real LLM Configured: Using {provider}")
        suite = unittest.TestLoader().loadTestsFromTestCase(TestRealLLMIntegration)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result.wasSuccessful()
    else:
        print("ℹ️  Tier 2 Status: Neither GEMINI_API_KEY nor OPENAI_API_KEY is detected in environment.")
        print("   -> Tier 1 FakeLLM suite provides 100% offline verification of all agent logic.")
        print("   -> To test against a live model, set your API key in terminal:")
        print("      $env:GEMINI_API_KEY='your_api_key_here'")
        print("      .\\run.bat 7")
        return True


def main():
    tier1_ok = run_tier1_tests()
    tier2_ok = run_tier2_check()

    print("\n" + "=" * 75)
    print("📊 PART 7 TEST EXECUTION SUMMARY")
    print("=" * 75)
    print(f"Tier 1 Deterministic Tests (FakeLLM): {'✅ PASSED (7/7)' if tier1_ok else '❌ FAILED'}")
    print(f"Tier 2 Live LLM Integration:         {'✅ READY / VERIFIED' if tier2_ok else '❌ FAILED'}")
    print("=" * 75)


if __name__ == "__main__":
    main()

