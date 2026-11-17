"""
Part 3 Demonstration Script: Fixing Errors When a Tool Fails (Self-Healing)
==========================================================================
File: agent/run_part3.py
Demonstrates:
1. Tool failure on Turn 1 (invalid argument types or misspelled tool name).
2. The runtime wrapping the Python exception into a structured error Observation.
3. The agent reading the error in Turn 2, reflecting in Thought, and self-correcting.
4. Successful execution on Turn 2 and completion with Final Answer on Turn 3.
"""

from __future__ import annotations

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from agent.loop import AgentLoop
    from agent.tools import default_registry
except ImportError:
    from loop import AgentLoop
    from tools import default_registry


# ============================================================================
# Simulated LLMs Demonstrating Self-Healing
# ============================================================================

class SelfHealingParamLLM:
    """
    Simulates an agent that accidentally passes a string instead of a list,
    receives the error Observation, explains the fix in Thought, and succeeds.
    """

    def __init__(self):
        self.turn = 0

    def __call__(self, prompt: str) -> str:
        self.turn += 1

        if self.turn == 1:
            # Mistake: passes fields as string "vendor, total_amount" instead of list
            return (
                "Thought: I need to extract the vendor and total amount from the invoice.\n"
                'Action: extract_structured_fields({"file_path": "invoice.pdf", "fields": "vendor, total_amount"})\n'
            )

        elif self.turn == 2:
            # Self-healing: reads the error observation and fixes the argument format to a list
            return (
                "Thought: The previous tool call failed because 'fields' was passed as a string instead of a list of strings. "
                "I will self-correct by passing a proper JSON list: [\"vendor\", \"total_amount\"].\n"
                'Action: extract_structured_fields({"file_path": "invoice.pdf", "fields": ["vendor", "total_amount"]})\n'
            )

        elif self.turn == 3:
            # Succeeded! Now outputs final answer
            return (
                "Thought: The tool succeeded on the second attempt and returned the extracted data.\n"
                "Final Answer: Extraction complete after self-healing: Vendor is Acme Global Supplies LLC and Total Amount is $1,353.13."
            )

        return "Final Answer: Done."


class SelfHealingTypoLLM:
    """
    Simulates an agent that mistypes a tool name (inspect_visual_doc),
    reads the suggested fix ("Did you mean 'inspect_visual_document'?"),
    and recovers immediately.
    """

    def __init__(self):
        self.turn = 0

    def __call__(self, prompt: str) -> str:
        self.turn += 1

        if self.turn == 1:
            # Mistake: typo in tool name
            return (
                "Thought: I want to inspect the document first.\n"
                'Action: inspect_visual_doc({"file_path": "claim_photo.jpg", "query": "Damage check"})\n'
            )

        elif self.turn == 2:
            # Self-healing: uses the suggestion provided in the error observation
            return (
                "Thought: The system reported that 'inspect_visual_doc' does not exist and suggested 'inspect_visual_document'. "
                "I will call inspect_visual_document with the correct parameters.\n"
                'Action: inspect_visual_document({"file_path": "claim_photo.jpg", "query": "Damage check"})\n'
            )

        elif self.turn == 3:
            return (
                "Thought: The visual inspection succeeded.\n"
                "Final Answer: The document was inspected successfully using the corrected tool name."
            )

        return "Final Answer: Done."


# ============================================================================
# Main Execution Demo
# ============================================================================

def main():
    print("=" * 75)
    print("DEMO: Part 3 - Fixing Errors When a Tool Fails (Self-Healing ReAct Loop)")
    print("=" * 75)

    # ------------------------------------------------------------------------
    # Scenario A: Parameter Error & Self-Correction
    # ------------------------------------------------------------------------
    print("\n" + "#" * 75)
    print("SCENARIO A: Parameter Type Error -> Observation -> Self-Correction")
    print("#" * 75)

    agent_a = AgentLoop(
        tools=default_registry,
        llm_caller=SelfHealingParamLLM(),
        max_steps=5,
        verbose=True,
    )

    result_a = agent_a.run("Extract vendor and total_amount from invoice.pdf")

    print("\n📋 Trajectory Analysis for Scenario A:")
    for step in result_a.trajectory:
        err_indicator = "❌ (Triggered Error)" if step.observation and step.observation.is_error else "✅ (Success)"
        print(f"  Step {step.step_number}: Tool = {step.action.tool_name if step.action else 'None'} {err_indicator}")
        if step.observation and step.observation.is_error:
            print(f"    -> Raw Error Observation Caught: {step.observation.content}")

    # ------------------------------------------------------------------------
    # Scenario B: Tool Name Typo & Suggestion Recovery
    # ------------------------------------------------------------------------
    print("\n" + "#" * 75)
    print("SCENARIO B: Tool Name Typo -> Suggestion -> Recovery")
    print("#" * 75)

    agent_b = AgentLoop(
        tools=default_registry,
        llm_caller=SelfHealingTypoLLM(),
        max_steps=5,
        verbose=True,
    )

    result_b = agent_b.run("Inspect claim_photo.jpg")

    print("\n📋 Trajectory Analysis for Scenario B:")
    for step in result_b.trajectory:
        err_indicator = "❌ (Triggered Error)" if step.observation and step.observation.is_error else "✅ (Success)"
        print(f"  Step {step.step_number}: Tool = {step.action.tool_name if step.action else 'None'} {err_indicator}")
        if step.observation and step.observation.is_error:
            print(f"    -> Raw Error Observation Caught: {step.observation.content}")

    print("\n" + "=" * 75)
    print("🎯 PART 3 SUMMARY: Both error scenarios recovered without crashing the process!")
    print("=" * 75)


if __name__ == "__main__":
    main()
