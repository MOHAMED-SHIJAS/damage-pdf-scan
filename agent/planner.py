"""
Part 5: A Plan Before It Acts
==============================
Module: agent/planner.py
Concept: Explicit pre-action planning engine in pure Python.
Forces the agent to formulate a multi-step execution plan with status markers
([TODO], [IN_PROGRESS], [COMPLETED], [FAILED]) before executing any tools.

How it works (for Demo Explanation):
------------------------------------
1. BEFORE ANY ACTION: The agent generates an initial numbered plan.
2. STEP TRACKING: As each tool runs, the plan marks the active step [IN_PROGRESS]
   and marks it [COMPLETED] when the tool returns success.
3. CONTEXT REINFORCEMENT: The updated plan is injected into every subsequent turn
   so the agent always knows its current position and what remains.
4. DYNAMIC ADJUSTMENT: If an error or unexpected observation occurs, the agent can
   revise or insert new steps dynamically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class PlanStep:
    """Represents a single step in the execution plan."""
    step_number: int
    description: str
    status: str = "TODO"  # "TODO", "IN_PROGRESS", "COMPLETED", "FAILED"

    def __str__(self) -> str:
        return f"{self.step_number}. [{self.status}] {self.description}"


@dataclass
class ExecutionPlan:
    """Manages the full list of steps in an agent's execution plan."""
    steps: List[PlanStep] = field(default_factory=list)

    def add_step(self, description: str, status: str = "TODO") -> PlanStep:
        """Adds a new step to the plan."""
        step_num = len(self.steps) + 1
        step = PlanStep(step_number=step_num, description=description, status=status)
        self.steps.append(step)
        return step

    def get_step(self, step_number: int) -> Optional[PlanStep]:
        """Retrieves a step by its number."""
        for s in self.steps:
            if s.step_number == step_number:
                return s
        return None

    def mark_in_progress(self, step_number: int) -> None:
        """Marks a step as currently in progress."""
        step = self.get_step(step_number)
        if step:
            step.status = "IN_PROGRESS"

    def mark_completed(self, step_number: int) -> None:
        """Marks a step as completed."""
        step = self.get_step(step_number)
        if step:
            step.status = "COMPLETED"

    def mark_failed(self, step_number: int, reason: str = "") -> None:
        """Marks a step as failed."""
        step = self.get_step(step_number)
        if step:
            step.status = "FAILED"
            if reason:
                step.description += f" (Failure: {reason})"

    def advance_next(self) -> Optional[PlanStep]:
        """
        Advances the first TODO step to IN_PROGRESS and returns it.
        Marks any previously IN_PROGRESS step as COMPLETED.
        """
        # Complete any in-progress step
        for s in self.steps:
            if s.status == "IN_PROGRESS":
                s.status = "COMPLETED"

        # Find next TODO
        for s in self.steps:
            if s.status == "TODO":
                s.status = "IN_PROGRESS"
                return s
        return None

    def is_all_completed(self) -> bool:
        """Checks if all steps in the plan are completed."""
        return all(s.status == "COMPLETED" for s in self.steps)

    def to_prompt_text(self) -> str:
        """Formats the current plan for injection into LLM prompts."""
        if not self.steps:
            return "No plan established."
        lines = [str(s) for s in self.steps]
        return "\n".join(lines)

    @classmethod
    def parse(cls, plan_text: str) -> ExecutionPlan:
        """
        Parses a numbered plan from raw LLM output text.
        Handles both '[TODO]' tagged lines and standard '1. ...' numbered lists.
        """
        plan = cls()
        lines = plan_text.strip().split("\n")
        step_pattern = re.compile(r"^\s*(\d+)[\.\)]\s*(?:\[(TODO|IN_PROGRESS|COMPLETED|FAILED)\])?\s*(.+)$", re.IGNORECASE)

        for line in lines:
            line = line.strip()
            match = step_pattern.match(line)
            if match:
                step_num = int(match.group(1))
                status = match.group(2).upper() if match.group(2) else "TODO"
                description = match.group(3).strip()
                plan.steps.append(
                    PlanStep(
                        step_number=step_num,
                        description=description,
                        status=status,
                    )
                )

        # Fallback if no numbered items matched
        if not plan.steps:
            for idx, line in enumerate(lines, 1):
                clean = line.lstrip("-* ").strip()
                if clean:
                    plan.steps.append(PlanStep(step_number=idx, description=clean, status="TODO"))

        return plan


# ============================================================================
# Planner Engine
# ============================================================================

PLANNER_SYSTEM_PROMPT = """You are the Pre-Action Planning Engine for a Vision & Multimodal Workflow Automation Agent.
Your job is to formulate an explicit, step-by-step numbered execution plan before the agent invokes any tools.

Available Tools:
{tool_descriptions}

Rules for Planning:
1. Break down the user's task into 3 to 5 logical, sequential steps.
2. Format each step on its own line: 1. [TODO] <description of step>
3. Order: Start with inspection/orientation, proceed to extraction, then validation, and finally system actions or summary.
4. Output ONLY the numbered plan. Do not execute any tools yet.
"""


class PlannerEngine:
    """Generates and maintains execution plans for agent workflows."""

    def __init__(self, llm_caller: Callable[[str], str]):
        self.llm_caller = llm_caller

    def create_plan(self, user_goal: str, tool_descriptions: str) -> ExecutionPlan:
        """
        Asks the model to formulate a numbered plan before taking actions.
        """
        prompt = (
            f"{PLANNER_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)}\n\n"
            f"User Task: {user_goal}\n"
            "Execution Plan:"
        )
        response = self.llm_caller(prompt).strip()
        plan = ExecutionPlan.parse(response)
        return plan

