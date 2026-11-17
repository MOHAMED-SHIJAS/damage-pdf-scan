"""
Part 1 Demonstration Script: The Core ReAct Loop
=================================================
File: agent/run_part1.py
Demonstrates: Think -> Act -> Look in pure Python with simulated LLM responses.
Run this script to see the Part 1 loop in action and verify its behavior.
"""

from __future__ import annotations

import json
from typing import Dict, Any

from loop import (
    AgentLoop,
    ReActParser,
    Action,
    Thought,
    Observation,
)


# ============================================================================
# 1. Sample Tools for Part 1 Demo
# ============================================================================

def inspect_visual_document(file_path: str, query: str) -> Dict[str, Any]:
    """Inspects an image or document, returns visual layout, text summary, and detected elements."""
    # Simulated inspection logic
    return {
        "status": "ok",
        "file": file_path,
        "query": query,
        "visual_findings": "Detected high-resolution scan of a commercial invoice from 'Acme Corp' with line items and subtotal/tax/total.",
        "quality": "good",
    }


def extract_structured_fields(file_path: str, fields: list) -> Dict[str, Any]:
    """Extracts targeted values into structured JSON with bounding box coordinates and confidence."""
    # Simulated field extraction logic
    return {
        "status": "ok",
        "extracted_data": {
            "vendor": "Acme Corp",
            "invoice_number": "INV-2024-001",
            "total_amount": 1450.00,
            "currency": "USD",
        },
        "bounding_boxes": {
            "invoice_number": [120, 45, 180, 60],
            "total_amount": [780, 520, 840, 540],
        },
        "confidence": 0.98,
    }


# ============================================================================
# 2. Simulated LLM Caller (Deterministic for Testing & Demo)
# ============================================================================

class SimulatedLLM:
    """
    Simulates a sequence of model completions following the ReAct pattern.
    This allows testing and demonstrating the exact Think -> Act -> Look loop
    without requiring API keys or network calls.
    """

    def __init__(self):
        self.call_count = 0

    def __call__(self, prompt: str) -> str:
        self.call_count += 1

        if self.call_count == 1:
            # Turn 1: Inspect the document first
            return (
                "Thought: I need to inspect the visual document to determine its layout and content.\n"
                'Action: inspect_visual_document({"file_path": "invoice_sample.png", "query": "Document type and header"})\n'
            )

        elif self.call_count == 2:
            # Turn 2: After observing the inspection, extract specific fields
            return (
                "Thought: The document is an invoice from Acme Corp. Now I will extract the invoice number, vendor, and total amount.\n"
                'Action: extract_structured_fields({"file_path": "invoice_sample.png", "fields": ["vendor", "invoice_number", "total_amount"]})\n'
            )

        elif self.call_count == 3:
            # Turn 3: Complete with Final Answer
            return (
                "Thought: I have extracted the required fields with high confidence (0.98).\n"
                "Final Answer: The document has been successfully processed. Vendor is Acme Corp, Invoice Number is INV-2024-001, and Total Amount is $1,450.00 USD."
            )

        else:
            return "Final Answer: Process finished."


# ============================================================================
# 3. Main Execution
# ============================================================================

def main():
    print("=" * 70)
    print("DEMO: Part 1 - The Core ReAct Loop (Think -> Act -> Look)")
    print("=" * 70)

    # 1. Setup tools dictionary
    tools = {
        "inspect_visual_document": inspect_visual_document,
        "extract_structured_fields": extract_structured_fields,
    }

    # 2. Instantiate Simulated LLM
    simulated_llm = SimulatedLLM()

    # 3. Create the AgentLoop
    agent = AgentLoop(
        tools=tools,
        llm_caller=simulated_llm,
        max_steps=5,
        verbose=True,
    )

    # 4. Run the Agent
    user_request = "Please process invoice_sample.png and extract the key financial fields."
    response = agent.run(user_request)

    # 5. Output Summary
    print("\n" + "=" * 70)
    print("📊 EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Success: {response.success}")
    print(f"Stop Reason: {response.stop_reason}")
    print(f"Total Steps Taken: {response.total_steps}")
    print(f"Recorded Trajectory Steps: {len(response.trajectory)}")
    print("\nStep-by-Step Trajectory JSON:")
    trajectory_json = json.dumps([s.to_dict() for s in response.trajectory], indent=2)
    print(trajectory_json)


if __name__ == "__main__":
    main()

