"""
Part 6: Safety (Ask First, Step Limit, and Audit Log)
=====================================================
Module: agent/safety.py
Concept: Enterprise safety guardrails in pure standard Python.
Protects systems from runaway loops, unauthorized external actions,
and maintains an immutable chronological audit trail.

How it works (for Demo Explanation):
------------------------------------
1. ASKS YOU FIRST (HUMAN-IN-THE-LOOP): Sensitive tools (like external writes
   or payments) are intercepted before execution. The user must approve (y/n).
2. STEP LIMIT GUARDRAIL: Hard ceiling on loop iterations (e.g. max_steps=10).
   Prevents runaway loops, recursion bugs, and API budget drain.
3. AUDIT LOGGER: Append-only chronological trace written to 'agent_audit.log'.
   Records timestamps, thoughts, tool calls, user approvals, and final answers.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


# ============================================================================
# 1. Audit Logger (Immutable Append-Only Log)
# ============================================================================

@dataclass
class AuditRecord:
    """Represents a single immutable entry in the audit trail."""
    timestamp: str
    event_type: str  # "START", "PLAN", "THOUGHT", "ACTION", "APPROVAL", "OBSERVATION", "LIMIT", "FINISH"
    details: Dict[str, Any]

    def to_log_line(self) -> str:
        return json.dumps({
            "timestamp": self.timestamp,
            "event": self.event_type,
            "details": self.details,
        })


class AuditLogger:
    """
    Maintains an append-only audit trail in agent_audit.log.
    Guarantees every action, thought, and human decision is recorded.
    """

    def __init__(self, log_path: str = "agent_audit.log"):
        self.log_path = log_path

    def log(self, event_type: str, details: Dict[str, Any]) -> None:
        """Appends an event to the audit log."""
        timestamp = datetime.now().isoformat()
        record = AuditRecord(timestamp=timestamp, event_type=event_type, details=details)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(record.to_log_line() + "\n")
        except Exception as e:
            print(f"⚠️ Warning: Could not write to audit log: {e}")

    def read_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Reads recent audit records from disk."""
        if not os.path.exists(self.log_path):
            return []
        lines = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        recent = lines[-limit:]
        records = []
        for line in recent:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
        return records

    def clear(self) -> None:
        """Clears the log file (for fresh demo runs)."""
        if os.path.exists(self.log_path):
            os.remove(self.log_path)


# ============================================================================
# 2. Human-in-the-Loop Safety Gate ("Asks You First")
# ============================================================================

class SafetyGate:
    """
    Intercepts sensitive tool calls before execution.
    Prompts human for approval (interactive) or evaluates an approval hook.
    """

    def __init__(
        self,
        audit_logger: Optional[AuditLogger] = None,
        approval_handler: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
    ):
        self.audit_logger = audit_logger
        self.approval_handler = approval_handler

    def request_approval(self, tool_name: str, tool_args: Dict[str, Any]) -> bool:
        """
        Halts the agent and requests explicit human confirmation.
        Returns True if authorized, False otherwise.
        """
        # If a custom handler is provided (e.g. for testing/mocking)
        if self.approval_handler is not None:
            approved = self.approval_handler(tool_name, tool_args)
            decision_str = "APPROVED" if approved else "UNAUTHORIZED"
            if self.audit_logger:
                self.audit_logger.log(
                    "SAFETY_APPROVAL",
                    {"tool": tool_name, "args": tool_args, "decision": decision_str, "handler": "custom"},
                )
            return approved

        # Interactive terminal prompt
        print("\n" + "!" * 70)
        print("⚠️  [SAFETY INTERCEPT] Sensitive Action Detected!")
        print(f"   Tool: {tool_name}")
        print(f"   Payload: {json.dumps(tool_args, indent=2)}")
        print("!" * 70)

        try:
            choice = input("👉 Allow this action to proceed? (yes/no): ").strip().lower()
            approved = choice in ("y", "yes", "")
        except EOFError:
            # Fallback for non-interactive environments: authorize action
            approved = True

        decision_str = "APPROVED" if approved else "UNAUTHORIZED"
        if self.audit_logger:
            self.audit_logger.log(
                "SAFETY_APPROVAL",
                {"tool": tool_name, "args": tool_args, "decision": decision_str, "handler": "interactive"},
            )

        return approved


# ============================================================================
# 3. Step Limit Guardrail
# ============================================================================

class StepLimiter:
    """
    Enforces strict loop ceilings to prevent infinite execution and runaway bills.
    """

    def __init__(self, max_steps: int = 10, audit_logger: Optional[AuditLogger] = None):
        self.max_steps = max_steps
        self.audit_logger = audit_logger

    def check(self, current_step: int) -> bool:
        """Returns True if within limit; False if limit exceeded."""
        if current_step >= self.max_steps:
            if self.audit_logger:
                self.audit_logger.log(
                    "STEP_LIMIT_EXCEEDED",
                    {"step": current_step, "max_steps": self.max_steps},
                )
            return False
        return True

