"""
Part 2 Demonstration Script: 2-4 Tools Registry & Execution
============================================================
File: agent/run_part2.py
Demonstrates:
1. ToolRegistry loading all 4 domain tools.
2. Generating LLM system prompt descriptions automatically.
3. Executing each of the 4 tools through the registry.
4. Running the AgentLoop from Part 1 using the real ToolRegistry.
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
    from agent.tools import (
        ToolRegistry,
        default_registry,
        inspect_visual_document,
        extract_structured_fields,
        validate_business_rules,
        dispatch_action_or_notification,
    )
except ImportError:
    from loop import AgentLoop
    from tools import (
        ToolRegistry,
        default_registry,
        inspect_visual_document,
        extract_structured_fields,
        validate_business_rules,
        dispatch_action_or_notification,
    )


class Part2SimulatedLLM:
    """Simulates an LLM using the 4 tools sequentially."""

    def __init__(self):
        self.step = 0

    def __call__(self, prompt: str) -> str:
        self.step += 1

        if self.step == 1:
            return (
                "Thought: I need to inspect the invoice file to understand its layout.\n"
                'Action: inspect_visual_document({"file_path": "inv_2024.pdf", "query": "Document layout and headers"})\n'
            )
        elif self.step == 2:
            return (
                "Thought: The file is an invoice. Now I will extract the financial fields.\n"
                'Action: extract_structured_fields({"file_path": "inv_2024.pdf", "fields": ["vendor", "subtotal", "tax_amount", "total_amount", "date", "po_number"]})\n'
            )
        elif self.step == 3:
            return (
                "Thought: I need to validate the extracted totals using the business rules.\n"
                'Action: validate_business_rules({"extracted_data": {"subtotal": 1250.0, "tax_amount": 103.13, "total_amount": 1353.13, "date": "2024-10-15", "po_number": "PO-99120"}, "rules": ["check_math_totals", "check_date_format", "require_po_number"]})\n'
            )
        elif self.step == 4:
            return (
                "Thought: Validation passed. I will now dispatch the approved data to the ERP system.\n"
                'Action: dispatch_action_or_notification({"action_type": "post_to_erp", "payload": {"vendor": "Acme Global Supplies LLC", "total": 1353.13, "status": "APPROVED"}})\n'
            )
        elif self.step == 5:
            return (
                "Thought: All tools executed successfully and transaction confirmed.\n"
                "Final Answer: Invoice inv_2024.pdf processed. Extracted total $1,353.13, verified arithmetic and PO number, and dispatched record to ERP system (TX-confirmed)."
            )
        return "Final Answer: Done."


def main():
    print("=" * 70)
    print("DEMO: Part 2 - 2 to 4 Tools, Clearly Described & Registered")
    print("=" * 70)

    # 1. Inspect registered tools in the registry
    print("\n📦 1. Registered Tools in Catalog:")
    print(f"Total Tools: {len(default_registry.list_tools())}")
    for tool_name in default_registry.list_tools():
        t = default_registry.get(tool_name)
        sensitivity = "⚠️ Sensitive" if t.is_sensitive else "🟢 Safe"
        print(f"  • {tool_name} [{sensitivity}]")

    # 2. Show auto-generated LLM documentation
    print("\n📝 2. Generated LLM Prompt Documentation:")
    print("-" * 50)
    print(default_registry.get_descriptions())
    print("-" * 50)

    # 3. Direct execution of each tool through registry
    print("\n⚡ 3. Direct Tool Executions via default_registry.execute():")

    # Tool 1
    t1_res = default_registry.execute(
        "inspect_visual_document",
        {"file_path": "sample_invoice.png", "query": "Document type"},
    )
    print(f"\n[Tool 1: inspect_visual_document] -> {t1_res['document_type']}")
    print(f"  Findings: {t1_res['visual_findings']}")

    # Tool 2
    t2_res = default_registry.execute(
        "extract_structured_fields",
        {"file_path": "sample_invoice.png", "fields": ["vendor", "total_amount", "po_number"]},
    )
    print(f"\n[Tool 2: extract_structured_fields] -> Mean Confidence: {t2_res['mean_confidence']}")
    print(f"  Extracted: {t2_res['extracted_data']}")
    print(f"  Bounding Boxes: {t2_res['bounding_boxes']}")

    # Tool 3
    t3_res = default_registry.execute(
        "validate_business_rules",
        {
            "extracted_data": {"subtotal": 1250.0, "tax_amount": 103.13, "total_amount": 1353.13, "date": "2024-10-15"},
            "rules": ["check_math_totals", "check_date_format"],
        },
    )
    print(f"\n[Tool 3: validate_business_rules] -> Valid: {t3_res['valid']}")
    print(f"  Passed Checks: {t3_res['passed']}")

    # Tool 4
    t4_res = default_registry.execute(
        "dispatch_action_or_notification",
        {"action_type": "post_to_erp", "payload": {"invoice_id": "INV-2024-8842", "status": "APPROVED"}},
    )
    print(f"\n[Tool 4: dispatch_action_or_notification] -> Status: {t4_res['status']}")
    print(f"  Confirmation ID: {t4_res['confirmation_id']}")

    # 4. End-to-end integration with Part 1 AgentLoop
    print("\n" + "=" * 70)
    print("🤖 4. End-to-End Agent Execution with All 4 Tools via AgentLoop")
    print("=" * 70)

    agent = AgentLoop(
        tools=default_registry,
        llm_caller=Part2SimulatedLLM(),
        max_steps=8,
        verbose=True,
    )

    result = agent.run("Process incoming invoice inv_2024.pdf through validation and post to ERP.")
    print("\n✅ Final Result Summary:")
    print(f"Status: {result.stop_reason}")
    print(f"Total Steps: {result.total_steps}")
    print(f"Final Answer: {result.final_answer}")


if __name__ == "__main__":
    main()

