"""
Part 4 Demonstration Script: Memory Between Tasks
==================================================
File: agent/run_part4.py
Demonstrates:
1. Task 1: The agent processes an invoice and learns a vendor-specific quirk
   ("Stark Industries uses DD/MM/YYYY dates and places the PO in the footer").
2. Persistence: The fact is saved to 'memory_store.json' on disk.
3. Process Restart: A fresh agent instance is created with no in-memory state.
4. Task 2: The user gives a new Stark Industries invoice WITHOUT specifying the quirk.
   The agent recalls the rule from memory_store.json and succeeds!
"""

from __future__ import annotations

import json
import os
from loop import AgentLoop
from memory import TaskMemoryStore
from tools import default_registry


# ============================================================================
# Simulated LLMs for Task 1 and Task 2
# ============================================================================

class Task1LearningLLM:
    """Simulates Task 1 where the agent learns a vendor rule and saves it."""

    def __init__(self):
        self.turn = 0

    def __call__(self, prompt: str) -> str:
        self.turn += 1

        if self.turn == 1:
            return (
                "Thought: I need to inspect the Stark Industries invoice and record its layout conventions.\n"
                'Action: inspect_visual_document({"file_path": "stark_inv_1001.pdf", "query": "Date format and PO location"})\n'
            )
        elif self.turn == 2:
            return (
                "Thought: I have learned that Stark Industries uses DD/MM/YYYY date formatting and places the PO in the footer.\n"
                'Action: extract_structured_fields({"file_path": "stark_inv_1001.pdf", "fields": ["vendor", "date", "po_number", "total_amount"]})\n'
            )
        elif self.turn == 3:
            return (
                "Thought: Extraction completed. I will conclude and record the vendor layout quirk.\n"
                "Final Answer: Successfully processed stark_inv_1001.pdf. Learned and recorded rule: Stark Industries invoices format dates as DD/MM/YYYY and place the PO in the bottom footer."
            )
        return "Final Answer: Done."


class Task2RecallingLLM:
    """
    Simulates Task 2 where the agent receives a new invoice from Stark Industries.
    It reads <MemoryContext> injected into the prompt and applies the learned rule
    without the user having to re-explain it!
    """

    def __init__(self):
        self.turn = 0

    def __call__(self, prompt: str) -> str:
        self.turn += 1

        if self.turn == 1:
            # Notice the agent's Thought directly references <MemoryContext>
            has_memory = "<MemoryContext>" in prompt and "Stark Industries" in prompt
            memory_note = "Using remembered rule from memory: Stark Industries uses DD/MM/YYYY dates and footer PO. " if has_memory else ""

            return (
                f"Thought: {memory_note}I will extract the fields applying the remembered format conventions.\n"
                'Action: extract_structured_fields({"file_path": "stark_inv_4412.pdf", "fields": ["vendor", "date", "po_number", "total_amount"]})\n'
            )
        elif self.turn == 2:
            return (
                "Thought: Extracted fields successfully. I will now validate the date and totals.\n"
                'Action: validate_business_rules({"extracted_data": {"date": "2024-10-15", "po_number": "PO-99120", "total_amount": 1353.13, "subtotal": 1250.0, "tax_amount": 103.13}, "rules": ["check_math_totals", "check_date_format", "require_po_number"]})\n'
            )
        elif self.turn == 3:
            return (
                "Thought: Validation passed using remembered vendor profile.\n"
                "Final Answer: Successfully processed new Stark Industries invoice #ST-4412 using cross-task memory without needing user guidance!"
            )
        return "Final Answer: Done."


# ============================================================================
# Main Execution Demo
# ============================================================================

def main():
    print("=" * 75)
    print("DEMO: Part 4 - Cross-Task Persistent Memory (memory_store.json)")
    print("=" * 75)

    storage_file = "memory_store.json"
    memory_store = TaskMemoryStore(storage_path=storage_file)
    memory_store.clear()  # Start fresh for clean demo

    # ------------------------------------------------------------------------
    # STEP 1: Execute Task 1 (Learning)
    # ------------------------------------------------------------------------
    print("\n" + "#" * 75)
    print("TASK 1: Process initial invoice & learn vendor quirk")
    print("#" * 75)

    agent_1 = AgentLoop(
        tools=default_registry,
        llm_caller=Task1LearningLLM(),
        memory=memory_store,
        max_steps=5,
        verbose=True,
    )

    t1_goal = "Process invoice stark_inv_1001.pdf from Stark Industries."
    res1 = agent_1.run(t1_goal)

    # Record the specific learned quirk into memory store
    memory_store.record_learned_fact(
        category="vendors",
        key="Stark Industries",
        value="Uses British DD/MM/YYYY date format; PO number is located in the bottom-left footer.",
    )

    print("\n💾 Task 1 Completed! State of 'memory_store.json' on disk:")
    with open(storage_file, "r", encoding="utf-8") as f:
        print(f.read())

    # ------------------------------------------------------------------------
    # STEP 2: Simulate Process Restart (Reloading from disk)
    # ------------------------------------------------------------------------
    print("\n" + "#" * 75)
    print("SIMULATING SYSTEM RESTART: Creating brand new Agent instance...")
    print("#" * 75)

    fresh_memory_store = TaskMemoryStore(storage_path=storage_file)

    # ------------------------------------------------------------------------
    # STEP 3: Execute Task 2 (Recalling learned fact without user prompt)
    # ------------------------------------------------------------------------
    print("\n" + "#" * 75)
    print("TASK 2: User provides NEW invoice ST-4412 from Stark Industries")
    print("(User does NOT mention date formats or PO locations)")
    print("#" * 75)

    agent_2 = AgentLoop(
        tools=default_registry,
        llm_caller=Task2RecallingLLM(),
        memory=fresh_memory_store,
        max_steps=5,
        verbose=True,
    )

    t2_goal = "Extract and validate new invoice stark_inv_4412.pdf from Stark Industries."
    res2 = agent_2.run(t2_goal)

    print("\n" + "=" * 75)
    print("🎯 PART 4 SUMMARY: Persistent memory survived restart and informed Task 2!")
    print("=" * 75)


if __name__ == "__main__":
    main()

