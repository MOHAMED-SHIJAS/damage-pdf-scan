"""
Part 8 (⭐ Bonus) Demonstration Script: Team of Agents
======================================================
File: agent/run_part8.py
Demonstrates:
1. Multi-agent architecture with 3 collaborating agents:
   - Orchestrator (Coordinates workflow & synthesizes reports)
   - Vision Analyst (Specialized in layout inspection & bounding boxes)
   - Compliance Auditor (Specialized in math checks & rule validation)
2. Typed message passing on the team message board.
3. End-to-end multi-agent execution on an invoice file.
"""

from __future__ import annotations

import json
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.team import MultiAgentTeam


def make_specialist_llm(role: str):
    """
    Returns a specialized simulated LLM completion function tailored to each agent role.
    """
    if role == "VisionAnalyst":
        def vision_caller(prompt: str) -> str:
            if "inspect_visual_document" not in prompt and "extract_structured_fields" not in prompt:
                return (
                    "Thought: As the Vision Analyst, I inspect visual features and extract fields.\n"
                    'Action: inspect_visual_document({"file_path": "multimodal_invoice.pdf", "query": "Layout and quality"})\n'
                )
            elif "extract_structured_fields" not in prompt:
                return (
                    "Thought: Layout confirmed. Now extracting coordinates and fields.\n"
                    'Action: extract_structured_fields({"file_path": "multimodal_invoice.pdf", "fields": ["vendor", "subtotal", "tax_amount", "total_amount", "date", "po_number"]})\n'
                )
            else:
                return (
                    "Thought: Visual extraction complete with high confidence.\n"
                    "Final Answer: Vision extraction finished. Extracted 6 fields with bounding boxes and 0.97 confidence."
                )
        return vision_caller

    elif role == "ComplianceAuditor":
        def auditor_caller(prompt: str) -> str:
            if "validate_business_rules" not in prompt:
                return (
                    "Thought: As the Compliance Auditor, I verify mathematical consistency and compliance rules.\n"
                    'Action: validate_business_rules({"extracted_data": {"subtotal": 1250.0, "tax_amount": 103.13, "total_amount": 1353.13, "date": "2024-10-15", "po_number": "PO-99120"}, "rules": ["check_math_totals", "check_date_format", "require_po_number"]})\n'
                )
            else:
                return (
                    "Thought: All business rules and arithmetic totals verified successfully.\n"
                    "Final Answer: Compliance audit passed. Zero violations found."
                )
        return auditor_caller

    else:
        def generic_caller(prompt: str) -> str:
            return "Final Answer: Done."
        return generic_caller


def main():
    print("=" * 75)
    print("DEMO: Part 8 (⭐ Bonus) - Team of Agents (Multi-Agent Collaboration)")
    print("=" * 75)

    team = MultiAgentTeam(llm_factory=make_specialist_llm, verbose=True)

    result = team.run_workflow(
        file_path="multimodal_invoice.pdf",
        user_instructions="Extract financial fields, audit math totals, and dispatch to ERP.",
    )

    print("\n" + "=" * 75)
    print("📋 SUMMARY OF AGENT-TO-AGENT MESSAGES EXCHANGED")
    print("=" * 75)
    for idx, msg in enumerate(team.message_board, 1):
        print(f"[{idx}] {msg.sender:<18} -> {msg.recipient:<18} | {msg.message_type}")
        print(f"    Payload: {json.dumps(msg.payload)[:90]}...")

    print("\n" + "=" * 75)
    print("🎯 PART 8 (⭐ BONUS) SUMMARY: Multi-agent team collaborated successfully!")
    print("=" * 75)


if __name__ == "__main__":
    main()

