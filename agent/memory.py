"""
Part 4: Memory Between Tasks
=============================
Module: agent/memory.py
Concept: Cross-task persistent memory store backed by a local JSON file.
Enables the agent to retain task summaries, vendor layout quirks,
and user preferences across separate runs and system restarts.

How it works (for Demo Explanation):
------------------------------------
1. IN-TASK WORKING MEMORY: Lives in RAM as the scratchpad during the active loop.
2. CROSS-TASK PERSISTENT MEMORY: Saved to 'memory_store.json'.
3. ON STARTUP: When a new task starts, the memory store retrieves relevant past
   facts and injects a `<MemoryContext>` block into the agent's system prompt.
4. ON COMPLETION: The agent automatically records a summary of the task and
   any new vendor quirks or facts learned into 'memory_store.json'.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class TaskRecord:
    """Record of a completed task run."""
    task_id: str
    goal: str
    outcome: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    facts_learned: Dict[str, Any] = field(default_factory=dict)


class TaskMemoryStore:
    """
    Persistent memory store backed by local JSON file storage.
    Tracks task execution history and learned domain facts.
    """

    def __init__(self, storage_path: str = "memory_store.json"):
        self.storage_path = storage_path
        self._data: Dict[str, Any] = {
            "task_history": [],
            "learned_facts": {},
            "vendor_profiles": {},
        }
        self.load()

    def load(self) -> None:
        """Loads memory state from disk if the file exists."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        self._data = json.loads(content)
            except Exception as e:
                print(f"⚠️ Warning: Could not load memory file ({e}). Initializing empty store.")
                self._data = {
                    "task_history": [],
                    "learned_facts": {},
                    "vendor_profiles": {},
                }
        else:
            self.save()

    def save(self) -> None:
        """Flushes the current in-memory state to disk atomically."""
        try:
            temp_path = f"{self.storage_path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
            if os.path.exists(self.storage_path):
                os.replace(temp_path, self.storage_path)
            else:
                os.rename(temp_path, self.storage_path)
        except Exception as e:
            print(f"⚠️ Warning: Failed to save memory store to disk: {e}")

    def record_learned_fact(self, category: str, key: str, value: Any) -> None:
        """
        Saves a learned fact into persistent memory.
        
        Args:
            category: Domain category (e.g. 'vendors', 'date_formats', 'user_preferences')
            key: Lookup identifier (e.g. 'Stark Industries')
            value: The fact or preference learned
        """
        if category not in self._data["learned_facts"]:
            self._data["learned_facts"][category] = {}
        self._data["learned_facts"][category][key] = value
        self.save()

    def get_learned_fact(self, category: str, key: str) -> Optional[Any]:
        """Retrieves a specific learned fact."""
        return self._data.get("learned_facts", {}).get(category, {}).get(key)

    def save_task_summary(
        self,
        task_id: str,
        goal: str,
        outcome: str,
        facts_learned: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Records a completed task run into history and updates facts."""
        record = TaskRecord(
            task_id=task_id,
            goal=goal,
            outcome=outcome,
            facts_learned=facts_learned or {},
        )
        self._data["task_history"].append(asdict(record))

        if facts_learned:
            for cat, facts in facts_learned.items():
                if isinstance(facts, dict):
                    for k, v in facts.items():
                        self.record_learned_fact(cat, k, v)
                else:
                    self.record_learned_fact("general", cat, facts)

        self.save()

    def get_recent_tasks(self, limit: int = 3) -> List[Dict[str, Any]]:
        """Returns the most recent completed task summaries."""
        history = self._data.get("task_history", [])
        return history[-limit:]

    def retrieve_context(self, query: str) -> str:
        """
        Finds relevant learned facts and recent task history to inject
        into the LLM system prompt under <MemoryContext>.
        """
        lines = []
        
        # 1. Include learned facts
        facts = self._data.get("learned_facts", {})
        if facts:
            lines.append("Learned Facts from Past Tasks:")
            for category, items in facts.items():
                for k, v in items.items():
                    # Simple keyword relevance filter
                    query_lower = query.lower()
                    if k.lower() in query_lower or category.lower() in query_lower or len(items) <= 5:
                        lines.append(f"- [{category.upper()}] {k}: {v}")

        # 2. Include recent tasks
        recent = self.get_recent_tasks(limit=2)
        if recent:
            lines.append("\nRecent Task History:")
            for task in recent:
                lines.append(f"- Task '{task['task_id']}': {task['goal']} -> Outcome: {task['outcome'][:80]}...")

        if not lines:
            return ""

        context_body = "\n".join(lines)
        return (
            "\n<MemoryContext>\n"
            "The following context was remembered from your previous tasks:\n"
            f"{context_body}\n"
            "</MemoryContext>\n"
        )

    def clear(self) -> None:
        """Resets the memory store (useful for tests)."""
        self._data = {
            "task_history": [],
            "learned_facts": {},
            "vendor_profiles": {},
        }
        self.save()

