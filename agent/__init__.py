"""
Vision & Multimodal Workflow Automation Agent (Plain Python)
Built strictly without third-party agent frameworks.
"""

from .loop import (
    Action,
    Thought,
    Observation,
    StepRecord,
    AgentResponse,
    ReActParser,
    AgentLoop,
    REACT_SYSTEM_PROMPT,
)
from .tools import (
    Tool,
    ToolRegistry,
    default_registry,
    tool,
    inspect_visual_document,
    extract_structured_fields,
    validate_business_rules,
    dispatch_action_or_notification,
    ALL_TOOLS,
)
from .memory import (
    TaskMemoryStore,
    TaskRecord,
)
from .planner import (
    PlanStep,
    ExecutionPlan,
    PlannerEngine,
)
from .safety import (
    SafetyGate,
    AuditLogger,
    AuditRecord,
    StepLimiter,
)
from .team import (
    MultiAgentTeam,
    AgentMessage,
)

__all__ = [
    # Part 1: Loop
    "Action",
    "Thought",
    "Observation",
    "StepRecord",
    "AgentResponse",
    "ReActParser",
    "AgentLoop",
    "REACT_SYSTEM_PROMPT",
    # Part 2: Tools
    "Tool",
    "ToolRegistry",
    "default_registry",
    "tool",
    "inspect_visual_document",
    "extract_structured_fields",
    "validate_business_rules",
    "dispatch_action_or_notification",
    "ALL_TOOLS",
    # Part 4: Memory
    "TaskMemoryStore",
    "TaskRecord",
    # Part 5: Planner
    "PlanStep",
    "ExecutionPlan",
    "PlannerEngine",
    # Part 6: Safety
    "SafetyGate",
    "AuditLogger",
    "AuditRecord",
    "StepLimiter",
    # Part 8: Team Bonus
    "MultiAgentTeam",
    "AgentMessage",
]
