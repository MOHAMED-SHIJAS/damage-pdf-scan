"""
Part 7: Tier 1 Test Suite - Deterministic Fake LLM Tests
=========================================================
File: tests/test_fake_llm.py
Concept: Tests the agent loop, tools, error recovery, memory, planning,
safety intercept, and step limits deterministically without API costs or network.

Compatible with both `python -m unittest` and `pytest`.
"""

import json
import os
import sys
import unittest
from typing import List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.loop import AgentLoop, AgentResponse, ReActParser
from agent.tools import ToolRegistry, default_registry
from agent.memory import TaskMemoryStore
from agent.planner import ExecutionPlan, PlannerEngine
from agent.safety import SafetyGate, AuditLogger, StepLimiter


class FakeLLM:
    """
    Deterministic mock LLM caller.
    Returns pre-scripted responses in sequence or based on prompt matching.
    """

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.call_history: List[str] = []

    def __call__(self, prompt: str) -> str:
        self.call_history.append(prompt)
        if not self.responses:
            raise IndexError("FakeLLM ran out of scripted responses!")
        return self.responses.pop(0)


class TestAgentDeterministic(unittest.TestCase):
    """Tier 1 offline deterministic test suite for all agent components."""

    def setUp(self):
        self.test_memory_path = "test_memory_store.json"
        self.test_audit_path = "test_agent_audit.log"
        if os.path.exists(self.test_memory_path):
            os.remove(self.test_memory_path)
        if os.path.exists(self.test_audit_path):
            os.remove(self.test_audit_path)

    def tearDown(self):
        if os.path.exists(self.test_memory_path):
            os.remove(self.test_memory_path)
        if os.path.exists(self.test_audit_path):
            os.remove(self.test_audit_path)

    # ------------------------------------------------------------------------
    # Test 1: Full Think -> Act -> Look Cycle (Part 1 & 2)
    # ------------------------------------------------------------------------
    def test_full_think_act_look_cycle(self):
        """Verifies normal execution completes across 3 turns."""
        scripted = [
            'Thought: Inspect layout.\nAction: inspect_visual_document({"file_path": "inv.pdf"})\n',
            'Thought: Extract fields.\nAction: extract_structured_fields({"file_path": "inv.pdf", "fields": ["vendor", "total_amount"]})\n',
            'Thought: All done.\nFinal Answer: Invoice processed successfully for Acme Global Supplies LLC.\n',
        ]
        fake_llm = FakeLLM(scripted)
        agent = AgentLoop(tools=default_registry, llm_caller=fake_llm, max_steps=5, verbose=False)
        response = agent.run("Process inv.pdf")

        self.assertTrue(response.success)
        self.assertEqual(response.stop_reason, "completed")
        self.assertEqual(response.total_steps, 3)
        self.assertIn("Invoice processed successfully", response.final_answer)
        self.assertEqual(len(response.trajectory), 3)

    # ------------------------------------------------------------------------
    # Test 2: Tool Failure & Self-Healing (Part 3)
    # ------------------------------------------------------------------------
    def test_tool_failure_and_self_healing(self):
        """Verifies tool error is caught and self-corrected on the next turn."""
        scripted = [
            # Turn 1: Mistake (passing string instead of list of fields)
            'Thought: Extracting.\nAction: extract_structured_fields({"file_path": "inv.pdf", "fields": "vendor, total"})\n',
            # Turn 2: Self-correction based on error observation
            'Thought: Tool returned error. Self-correcting to a list.\nAction: extract_structured_fields({"file_path": "inv.pdf", "fields": ["vendor", "total_amount"]})\n',
            # Turn 3: Final answer
            'Thought: Extracted.\nFinal Answer: Recovered and extracted $1,353.13.\n',
        ]
        fake_llm = FakeLLM(scripted)
        agent = AgentLoop(tools=default_registry, llm_caller=fake_llm, max_steps=5, verbose=False)
        response = agent.run("Extract invoice")

        self.assertTrue(response.success)
        # Verify step 1 had an error observation and step 2 succeeded
        self.assertTrue(response.trajectory[0].observation.is_error)
        self.assertIn("ValueError", response.trajectory[0].observation.content)
        self.assertFalse(response.trajectory[1].observation.is_error)
        self.assertIn("Acme Global Supplies LLC", response.trajectory[1].observation.content)

    # ------------------------------------------------------------------------
    # Test 3: Pre-Action Planning Engine (Part 5)
    # ------------------------------------------------------------------------
    def test_pre_action_planning(self):
        """Verifies PlannerEngine creates a plan and tracks steps to completion."""
        scripted = [
            # Call 0: Plan generation by PlannerEngine
            "1. [TODO] Inspect document\n2. [TODO] Extract fields\n3. [TODO] Dispatch\n",
            # Turn 1
            'Thought: Step 1.\nAction: inspect_visual_document({"file_path": "test.png"})\n',
            # Turn 2
            'Thought: Step 2.\nAction: extract_structured_fields({"file_path": "test.png", "fields": ["vendor"]})\n',
            # Turn 3
            'Thought: Plan finished.\nFinal Answer: Plan executed completely.\n',
        ]
        fake_llm = FakeLLM(scripted)
        agent = AgentLoop(
            tools=default_registry,
            llm_caller=fake_llm,
            enable_planning=True,
            max_steps=5,
            verbose=False,
        )
        response = agent.run("Execute workflow")

        self.assertTrue(response.success)
        self.assertIsNotNone(response.plan)
        self.assertTrue(response.plan.is_all_completed())
        self.assertEqual(len(response.plan.steps), 3)

    # ------------------------------------------------------------------------
    # Test 4: Safety Gate Interception & Human Approval (Part 6)
    # ------------------------------------------------------------------------
    def test_safety_gate_approval(self):
        """Verifies sensitive tool is intercepted and human approval authorizes execution."""
        scripted = [
            'Thought: Posting payment.\nAction: dispatch_action_or_notification({"action_type": "post_to_erp", "payload": {"amount": 5000}})\n',
            'Thought: Payment dispatch was authorized and completed.\nFinal Answer: Transaction approved and finished successfully.\n',
        ]
        fake_llm = FakeLLM(scripted)

        # Mock handler that approves execution
        safety_gate = SafetyGate(approval_handler=lambda tool, args: True)
        agent = AgentLoop(
            tools=default_registry,
            llm_caller=fake_llm,
            safety_gate=safety_gate,
            max_steps=5,
            verbose=False,
        )
        response = agent.run("Post payment")

        self.assertTrue(response.success)
        self.assertIn("success", response.trajectory[0].observation.content)
        self.assertIn("Transaction approved", response.final_answer)

    # ------------------------------------------------------------------------
    # Test 5: Step Limit Guardrail (Part 6)
    # ------------------------------------------------------------------------
    def test_step_limit_guardrail(self):
        """Verifies agent halts at max_steps if the model loops infinitely."""
        # Provide 10 repeated thoughts with no Final Answer
        scripted = [
            'Thought: Still thinking...\nAction: inspect_visual_document({"file_path": "x.png"})\n'
            for _ in range(10)
        ]
        fake_llm = FakeLLM(scripted)
        agent = AgentLoop(
            tools=default_registry,
            llm_caller=fake_llm,
            max_steps=3,  # Hard ceiling of 3 steps
            verbose=False,
        )
        response = agent.run("Infinite loop task")

        self.assertFalse(response.success)
        self.assertEqual(response.stop_reason, "max_steps_exceeded")
        self.assertEqual(response.total_steps, 3)

    # ------------------------------------------------------------------------
    # Test 6: Cross-Task Memory Persistence (Part 4)
    # ------------------------------------------------------------------------
    def test_cross_task_memory_persistence(self):
        """Verifies memory store persists facts to disk and reloads them into prompt."""
        mem = TaskMemoryStore(storage_path=self.test_memory_path)
        mem.record_learned_fact("vendors", "Acme", "Always uses net-30 terms")

        # Create a second fresh store pointing to the same file
        fresh_mem = TaskMemoryStore(storage_path=self.test_memory_path)
        fact = fresh_mem.get_learned_fact("vendors", "Acme")
        self.assertEqual(fact, "Always uses net-30 terms")

        # Verify context injection
        context = fresh_mem.retrieve_context("Process invoice from Acme")
        self.assertIn("<MemoryContext>", context)
        self.assertIn("Always uses net-30 terms", context)

    # ------------------------------------------------------------------------
    # Test 7: Document Scanner, Quality Enhancement & Categorization
    # ------------------------------------------------------------------------
    def test_document_scan_quality_enhancement_and_categorization(self):
        """Verifies scanner extracts genuine details, enhances quality, and never fabricates mock fields."""
        import shutil
        import tempfile
        from agent.scanner import DocumentScanner, default_scanner
        from agent.tools import (
            inspect_visual_document,
            extract_structured_fields,
            validate_business_rules,
            enhance_low_quality_details,
            categorize_extracted_details,
            scanner_registry,
        )

        tmp_dir = tempfile.mkdtemp(prefix="prd_scan_test_")
        try:
            scanner = DocumentScanner(uploads_dir=tmp_dir)

            # 1) Create a synthetic messy TXT file with OCR-like glitches and real metadata
            txt_path = os.path.join(tmp_dir, "note_sample.txt")
            with open(txt_path, "w", encoding="utf-8") as tf:
                tf.write("S.N0. 1\nRegistration No: GGAZSAIOII\nAuthor: Priya Nair\nEvent Date: 2025-09-12\n")

            res_txt = scanner.scan_and_extract(txt_path)
            self.assertTrue(res_txt.quality_report.is_enhanced)
            self.assertGreaterEqual(res_txt.quality_report.enhanced_score, res_txt.quality_report.original_score)
            self.assertGreater(len(res_txt.categorized_details), 0)
            self.assertGreater(len(res_txt.quality_report.before_after_repairs), 0)

            flat_json = json.dumps(res_txt.categorized_details)
            self.assertNotIn("Acme Global Supplies", flat_json)
            self.assertIn("Priya Nair", flat_json)

            # 2) Verify dedicated enhance_low_quality_details and categorize_extracted_details tools
            enh_out = scanner_registry.execute("enhance_low_quality_details", {"file_path": txt_path})
            self.assertEqual(enh_out["status"], "ok")
            self.assertGreater(enh_out["quality_score_enhanced"], enh_out["quality_score_original"])
            self.assertGreater(len(enh_out["before_after_repairs"]), 0)

            cat_out = scanner_registry.execute("categorize_extracted_details", {"file_path": txt_path})
            self.assertEqual(cat_out["status"], "ok")
            self.assertTrue(cat_out["zero_mock_fabrication_verified"])
            self.assertGreaterEqual(cat_out["total_categories"], 3)
            self.assertIn("Priya Nair", json.dumps(cat_out["categorized_details"]))

            # 2b) Verify full planned AgentLoop runs Step 1 inspect -> Step 2 enhance -> Step 3 categorize -> Step 4 validate
            from agent.scanner import ScanAwareAgentLLM
            loop_agent = AgentLoop(
                tools=scanner_registry,
                llm_caller=ScanAwareAgentLLM(txt_path, res_txt),
                enable_planning=True,
                max_steps=6,
                verbose=False,
            )
            loop_res = loop_agent.run(f"Scan, enhance and categorize {txt_path}")
            self.assertTrue(loop_res.success)
            self.assertIsNotNone(loop_res.plan)
            self.assertTrue(loop_res.plan.is_all_completed())
            tools_called = [s.action.tool_name for s in loop_res.trajectory if s.action]
            self.assertEqual(
                tools_called,
                [
                    "inspect_visual_document",
                    "enhance_low_quality_details",
                    "categorize_extracted_details",
                    "validate_business_rules",
                ],
            )

            # 3) Verify extract_structured_fields never leaks mock Acme data for real files on disk
            tool_res = extract_structured_fields(txt_path, ["vendor", "subtotal", "author"])
            self.assertIsNone(tool_res["extracted_data"]["vendor"])
            self.assertIsNone(tool_res["extracted_data"]["subtotal"])

            # 4) Verify domain-agnostic business rule validation passes cleanly
            val_out = validate_business_rules(
                {
                    "author": "Priya Nair",
                    "quality_score_original": res_txt.quality_report.original_score,
                    "quality_score_enhanced": res_txt.quality_report.enhanced_score,
                    "repairs_applied": len(res_txt.quality_report.enhancements_applied),
                    "categories_count": len(res_txt.categorized_details),
                },
                ["check_extracted_authenticity", "check_quality_enhancement", "check_category_completeness"],
            )
            self.assertTrue(val_out["valid"], f"Unexpected violations: {val_out['violations']}")

            # 4) If real PDF uploads/gx30 3.pdf is present, verify zero invoice contamination
            if os.path.exists("uploads/gx30 3.pdf"):
                inspect_out = inspect_visual_document("uploads/gx30 3.pdf", "Check title and author")
                self.assertEqual(inspect_out["status"], "ok")
                self.assertNotIn("Acme Global Supplies", json.dumps(inspect_out))
                gx_fields = extract_structured_fields("uploads/gx30 3.pdf", ["vendor", "subtotal", "document_author", "document_title"])
                self.assertIsNone(gx_fields["extracted_data"]["vendor"])
                self.assertIsNone(gx_fields["extracted_data"]["subtotal"])
                self.assertEqual(gx_fields["extracted_data"]["document_author"], "Lijas")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------------
    # Test 8: Multi-Format CSV & JSON Enhancement and Categorization
    # ------------------------------------------------------------------------
    def test_csv_and_json_enhancement_and_categorization(self):
        """Verifies CSV dirty-cell/currency cleaning and JSON hierarchical categorization."""
        import shutil
        import tempfile
        from agent.scanner import DocumentScanner

        tmp_dir = tempfile.mkdtemp(prefix="prd_fmt_test_")
        try:
            scanner = DocumentScanner(uploads_dir=tmp_dir)

            # Dirty CSV test
            csv_path = os.path.join(tmp_dir, "expenses.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as cf:
                cf.write("  Vendor , Amount , Date \n")
                cf.write("  CloudHost Inc. , \"$1,250.50\" , 2025-10-01 \n")
                cf.write("  DomainReg LLC ,  , 2025-10-02 \n")
                cf.write(" , , \n")

            res_csv = scanner.scan_and_extract(csv_path)
            self.assertEqual(res_csv.format_category, "CSV Spreadsheet")
            self.assertTrue(res_csv.quality_report.is_enhanced)
            self.assertGreater(res_csv.quality_report.enhanced_score, res_csv.quality_report.original_score)
            self.assertIsNotNone(res_csv.quality_report.enhanced_file_path)
            self.assertTrue(os.path.exists(res_csv.quality_report.enhanced_file_path))
            self.assertNotIn("Acme Global Supplies", json.dumps(res_csv.categorized_details))

            # Structured JSON test
            json_path = os.path.join(tmp_dir, "record.json")
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(
                    {
                        "candidate_name": "Ananya Menon",
                        "issue_date": "2025-11-01",
                        "total_fee": "$420.00",
                        "notes": None,
                    },
                    jf,
                )

            res_json = scanner.scan_and_extract(json_path)
            self.assertEqual(res_json.format_category, "JSON Document")
            self.assertGreater(len(res_json.categorized_details), 0)
            json_dump = json.dumps(res_json.categorized_details)
            self.assertIn("Ananya Menon", json_dump)
            self.assertNotIn("Acme Global Supplies", json_dump)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()


