"""
Part 5 Demonstration Script: A Plan Before It Acts
==================================================
File: agent/run_part5.py
Demonstrates:
1. Pre-action planning: The agent generates an explicit numbered plan before executing tools.
2. Step tracking: In each turn, the agent references the plan and transitions
   steps from [TODO] -> [IN_PROGRESS] -> [COMPLETED].
3. Structured completion: Final response includes the completed execution plan.
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
    from agent.planner import ExecutionPlan
except ImportError:
    from loop import AgentLoop
    from tools import default_registry
    from planner import ExecutionPlan


class PlanningAwareLLM:
    """
    Simulates an LLM that:
    1. First produces a numbered plan when called by PlannerEngine.
    2. Then executes the plan step-by-step in the ReAct loop.
    """

    def __init__(self):
        self.call_count = 0

    def __call__(self, prompt: str) -> str:
        self.call_count += 1

        # Call 1: PlannerEngine prompt asking for the initial plan
        if "Pre-Action Planning Engine" in prompt:
            return (
                "1. [TODO] Inspect visual document for layout, resolution, and header structure.\n"
                "2. [TODO] Extract vendor, invoice number, and financial totals with bounding boxes.\n"
                "3. [TODO] Validate arithmetic consistency (subtotal + tax == total) and PO number.\n"
                "4. [TODO] Dispatch approved invoice payload to the ERP system."
            )

        # Turn 1 of ReAct Loop (Step 1 of Plan)
        elif self.call_count == 2:
            return (
                "Thought: Following Step 1 of my plan, I will inspect Q4_inv.png to understand the document layout.\n"
                'Action: inspect_visual_document({"file_path": "Q4_inv.png", "query": "Document layout and headers"})\n'
            )

        # Turn 2 of ReAct Loop (Step 2 of Plan)
        elif self.call_count == 3:
            return (
                "Thought: Step 1 completed. Moving to Step 2: extracting vendor, totals, and PO number into structured JSON.\n"
                'Action: extract_structured_fields({"file_path": "Q4_inv.png", "fields": ["vendor", "subtotal", "tax_amount", "total_amount", "date", "po_number"]})\n'
            )

        # Turn 3 of ReAct Loop (Step 3 of Plan)
        elif self.call_count == 4:
            return (
                "Thought: Step 2 completed. Moving to Step 3: validating mathematical totals and checking the PO number.\n"
                'Action: validate_business_rules({"extracted_data": {"subtotal": 1250.0, "tax_amount": 103.13, "total_amount": 1353.13, "date": "2024-10-15", "po_number": "PO-99120"}, "rules": ["check_math_totals", "check_date_format", "require_po_number"]})\n'
            )

        # Turn 4 of ReAct Loop (Step 4 of Plan)
        elif self.call_count == 5:
            return (
                "Thought: Step 3 completed. Moving to Step 4: dispatching the verified data to ERP.\n"
                'Action: dispatch_action_or_notification({"action_type": "post_to_erp", "payload": {"vendor": "Acme Global Supplies LLC", "total": 1353.13, "status": "APPROVED"}})\n'
            )

        # Turn 5: Final Answer
        else:
            return (
                "Thought: All 4 steps of my plan are completed successfully.\n"
                "Final Answer: All tasks in the execution plan completed. Q4_inv.png inspected, fields extracted with bounding boxes, totals mathematically verified, and record posted to ERP (TX-confirmed)."
            )


def main():
    print("=" * 75)
    print("DEMO: Part 5 - A Plan Before It Acts (Explicit Planning Engine)")
    print("=" * 75)

    llm = PlanningAwareLLM()

    agent = AgentLoop(
        tools=default_registry,
        llm_caller=llm,
        enable_planning=True,  # Enables Part 5 Pre-Action Planner
        max_steps=6,
        verbose=True,
    )

    user_goal = "Process Q4_inv.png, validate financial totals, and post to ERP."
    response = agent.run(user_goal)

    print("\n" + "=" * 75)
    print("📊 FINAL EXECUTION PLAN STATE")
    print("=" * 75)
    if response.plan:
        for step in response.plan.steps:
            icon = "✅" if step.status == "COMPLETED" else "⏳"
            print(f"  {icon} Step {step.step_number}: [{step.status}] {step.description}")

    print(f"\nFinal Answer: {response.final_answer}")
    print(f"Total Steps Taken: {response.total_steps}")
    print(f"All Steps Completed: {response.plan.is_all_completed() if response.plan else False}")


if __name__ == "__main__":
    main()

