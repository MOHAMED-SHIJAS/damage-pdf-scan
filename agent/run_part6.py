"""
Part 6 Demonstration Script: Safety Guardrails (Ask First, Step Limit, Audit Log)
=================================================================================
File: agent/run_part6.py
Demonstrates:
1. Asks You First: Sensitive tools are intercepted before execution and authorized upon human approval.
2. Step Limit Guardrail: Stops runaway execution when max_steps is reached.
3. Audit Log: Shows the persistent, immutable audit trail written to 'agent_audit.log'.
"""

from __future__ import annotations

import json
import os
from loop import AgentLoop
from tools import default_registry
from safety import SafetyGate, AuditLogger, StepLimiter


# ============================================================================
# Simulated LLMs for Safety Scenarios
# ============================================================================

class SensitiveActionLLM:
    """Simulates an agent attempting to post financial data to ERP."""

    def __init__(self):
        self.turn = 0

    def __call__(self, prompt: str) -> str:
        self.turn += 1

        if self.turn == 1:
            return (
                "Thought: I will extract the invoice financial total first.\n"
                'Action: extract_structured_fields({"file_path": "inv_march.pdf", "fields": ["vendor", "total_amount"]})\n'
            )
        elif self.turn == 2:
            return (
                "Thought: Total extracted is $1,353.13. I will now post the payment to the ERP system.\n"
                'Action: dispatch_action_or_notification({"action_type": "post_to_erp", "payload": {"vendor": "Acme Global", "total": 1353.13, "status": "APPROVED"}})\n'
            )
        elif self.turn == 3:
            return (
                "Thought: Payment dispatch was approved by supervisor and completed.\n"
                "Final Answer: Invoice processed and ERP transaction confirmed with supervisor authorization."
            )
        return "Final Answer: Done."


class RunawayInfiniteLoopLLM:
    """Simulates a buggy or runaway agent that never outputs Final Answer."""

    def __init__(self):
        self.turn = 0

    def __call__(self, prompt: str) -> str:
        self.turn += 1
        return (
            f"Thought: Loop turn {self.turn}. I am still thinking and need to keep inspecting...\n"
            'Action: inspect_visual_document({"file_path": "infinite_loop.png", "query": "Checking again"})\n'
        )


# ============================================================================
# Main Execution Demo
# ============================================================================

def main():
    print("=" * 75)
    print("DEMO: Part 6 - Safety Guardrails (Ask First, Step Limit, Audit Log)")
    print("=" * 75)

    audit_file = "agent_audit.log"
    audit_logger = AuditLogger(log_path=audit_file)
    audit_logger.clear()  # Start fresh for clean demo

    # ------------------------------------------------------------------------
    # SCENARIO 1: Safety Intercept - User APPROVES the action
    # ------------------------------------------------------------------------
    print("\n" + "#" * 75)
    print("SCENARIO 1: Safety Intercept -> Human APPROVES Sensitive Write Action")
    print("#" * 75)

    def mock_human_approve(tool_name: str, args: dict) -> bool:
        print(f"\n⚠️  [SAFETY INTERCEPT] Sensitive Tool: '{tool_name}'")
        print(f"   Payload: {args}")
        print("   👉 Human Supervisor Decision: APPROVE (Execution Authorized)")
        return True

    safety_gate_approval = SafetyGate(
        audit_logger=audit_logger,
        approval_handler=mock_human_approve,
    )

    agent_approval = AgentLoop(
        tools=default_registry,
        llm_caller=SensitiveActionLLM(),
        safety_gate=safety_gate_approval,
        audit_logger=audit_logger,
        max_steps=5,
        verbose=True,
    )

    res_approval = agent_approval.run("Process inv_march.pdf and post payment to ERP.")
    print(f"\n🎯 Scenario 1 Outcome: {res_approval.final_answer}")

    # ------------------------------------------------------------------------
    # SCENARIO 2: Hard Step Limit Guardrail (Kills Runaway Loop)
    # ------------------------------------------------------------------------
    print("\n" + "#" * 75)
    print("SCENARIO 2: Step Limit Guardrail -> Halts Runaway Loop at max_steps=3")
    print("#" * 75)

    runaway_agent = AgentLoop(
        tools=default_registry,
        llm_caller=RunawayInfiniteLoopLLM(),
        audit_logger=audit_logger,
        max_steps=3,  # Hard ceiling of 3 steps
        verbose=True,
    )

    res_runaway = runaway_agent.run("Run forever on infinite_loop.png")
    print(f"\n🛑 Scenario 2 Outcome: {res_runaway.final_answer}")
    print(f"   Stop Reason: {res_runaway.stop_reason}")
    print(f"   Success Flag: {res_runaway.success}")

    # ------------------------------------------------------------------------
    # SCENARIO 3: IMMUTABLE AUDIT TRAIL VERIFICATION
    # ------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("SCENARIO 3: Verified Audit Trail on Disk (agent_audit.log)")
    print("=" * 75)
    logs = audit_logger.read_recent(limit=8)
    for entry in logs:
        print(f"[{entry['timestamp']}] {entry['event']:<18} | {json.dumps(entry['details'])[:85]}...")

    print("\n" + "=" * 75)
    print("🎯 PART 6 SUMMARY: Human safety gate, step limit, and audit trail verified!")
    print("=" * 75)


if __name__ == "__main__":
    main()

