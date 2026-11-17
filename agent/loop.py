"""
Part 1: The Core Loop (Think -> Act -> Look)
==============================================
Module: agent/loop.py
Concept: Pure Python implementation of the ReAct (Reasoning + Acting) loop.
No agent libraries (LangChain, CrewAI, AutoGen, etc.) are used.

How it works (for Demo Explanation):
------------------------------------
1. THINK: The agent analyzes the user prompt, available tools, and past
   observations to decide its next move. It outputs its internal reasoning.
2. ACT: The agent selects an available tool and generates structured arguments
   (e.g., Action: inspect_visual_document({"file_path": "invoice.png"})).
3. LOOK: The runtime parses the action, executes the Python function, and feeds
   the return value back to the agent as an "Observation".
4. REPEAT: The loop continues until the agent outputs "Final Answer: ...".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


# ============================================================================
# 1. Core Data Structures
# ============================================================================

@dataclass
class Thought:
    """Represents the agent's internal reasoning step."""
    content: str

    def __str__(self) -> str:
        return f"Thought: {self.content}"

    def __getitem__(self, item: Any) -> Any:
        return self.content[item]

    def __len__(self) -> int:
        return len(self.content)


@dataclass
class Action:
    """Represents a tool call selected by the agent."""
    tool_name: str
    tool_args: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"Action: {self.tool_name}({json.dumps(self.tool_args)})"


@dataclass
class Observation:
    """Represents the output returned by the environment/tool after execution."""
    content: str
    is_error: bool = False

    def __str__(self) -> str:
        status_tag = "[ERROR] " if self.is_error else ""
        return f"Observation: {status_tag}{self.content}"


@dataclass
class StepRecord:
    """Tracks a single complete turn of (Think -> Act -> Look)."""
    step_number: int
    thought: Optional[Thought] = None
    action: Optional[Action] = None
    observation: Optional[Observation] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step_number,
            "thought": self.thought.content if self.thought else None,
            "action": {
                "tool": self.action.tool_name,
                "args": self.action.tool_args,
            } if self.action else None,
            "observation": self.observation.content if self.observation else None,
            "is_error": self.observation.is_error if self.observation else False,
        }


@dataclass
class AgentResponse:
    """Final output returned by the agent upon loop completion."""
    final_answer: str
    success: bool
    total_steps: int
    trajectory: List[StepRecord] = field(default_factory=list)
    stop_reason: str = "completed"  # "completed", "max_steps_exceeded", "error"
    plan: Optional[Any] = None  # ExecutionPlan from Part 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_answer": self.final_answer,
            "success": self.success,
            "total_steps": self.total_steps,
            "stop_reason": self.stop_reason,
            "trajectory": [s.to_dict() for s in self.trajectory],
            "plan": self.plan.to_dict() if self.plan and hasattr(self.plan, "to_dict") else None,
        }


# ============================================================================
# 2. System Prompt Template
# ============================================================================

REACT_SYSTEM_PROMPT = """You are a helpful Vision & Multimodal Workflow Automation Agent.
You solve tasks by executing actions one step at a time in a loop of Thought, Action, and Observation.

Use the following format strictly:

Question: The input task you must solve
Thought: Your reasoning about what to do next
Action: tool_name({{"arg_name": "arg_value"}})
Observation: The result of the action (provided by the system)
... (this Thought/Action/Observation can repeat up to N times)
Thought: I now have enough information to answer
Final Answer: The final response to the original question

Rules:
1. Only choose one action per turn.
2. The Action line MUST be formatted as: tool_name({{"key": "value"}}) with valid JSON inside the parentheses.
3. When you have completed the task or have the final answer, output 'Final Answer: ' followed by your response. Do not output an Action on the same turn as Final Answer.
4. If a tool returns an error in the Observation, analyze what went wrong in your next Thought and adjust your arguments or try another approach.

Available Tools:
{tool_descriptions}
"""


# ============================================================================
# 3. Robust Pure-Python ReAct Parser
# ============================================================================

class ReActParser:
    """
    Parses LLM output text into Thought, Action, and Final Answer.
    Uses regex and JSON decoding in pure standard Python.
    """

    # Regex patterns for parsing ReAct markers
    THOUGHT_PATTERN = re.compile(r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|$)", re.DOTALL | re.IGNORECASE)
    FINAL_ANSWER_PATTERN = re.compile(r"Final Answer:\s*(.*)", re.DOTALL | re.IGNORECASE)
    # Matches start of Action: tool_name(
    ACTION_HEAD_PATTERN = re.compile(r"Action:\s*([a-zA-Z0-9_]+)\s*\(", re.DOTALL | re.IGNORECASE)
    # Fallback: Action: tool_name \n { ... }
    ACTION_JSON_BLOCK_PATTERN = re.compile(r"Action:\s*([a-zA-Z0-9_]+)\s*\n\s*(\{.*\})", re.DOTALL | re.IGNORECASE)

    @classmethod
    def parse_thought(cls, text: str) -> Optional[Thought]:
        """Extracts the reasoning string from 'Thought: <content>'."""
        match = cls.THOUGHT_PATTERN.search(text)
        if match:
            thought_text = match.group(1).strip()
            if thought_text:
                return Thought(content=thought_text)
        return None

    @classmethod
    def parse_final_answer(cls, text: str) -> Optional[str]:
        """Checks if the model decided it is done and emitted 'Final Answer: <content>'."""
        match = cls.FINAL_ANSWER_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        return None

    @classmethod
    def parse_action(cls, text: str) -> Optional[Action]:
        """
        Extracts tool_name and parses its JSON arguments.
        Uses string-aware balanced parenthesis matching so parentheses inside
        JSON values like '(Autonomous)' or '(5.21 x 5.21 in)' never truncate args.
        """
        head = cls.ACTION_HEAD_PATTERN.search(text)
        if head:
            tool_name = head.group(1).strip()
            start_idx = head.end()
            depth = 1
            in_quote: Optional[str] = None
            escaped = False
            end_idx: Optional[int] = None

            for idx in range(start_idx, len(text)):
                ch = text[idx]
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    if in_quote is not None:
                        escaped = True
                    continue
                if ch in ('"', "'"):
                    if in_quote is None:
                        in_quote = ch
                    elif in_quote == ch:
                        in_quote = None
                    continue
                if in_quote is None:
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            end_idx = idx
                            break

            if end_idx is not None:
                args_str = text[start_idx:end_idx].strip()
                return cls._decode_action_args(tool_name, args_str)

        # Fallback pattern: tool_name \n { JSON }
        match_fallback = cls.ACTION_JSON_BLOCK_PATTERN.search(text)
        if match_fallback:
            tool_name = match_fallback.group(1).strip()
            args_str = match_fallback.group(2).strip()
            return cls._decode_action_args(tool_name, args_str)

        return None

    @classmethod
    def _decode_action_args(cls, tool_name: str, args_str: str) -> Action:
        """Safely parses arguments into a Python dictionary."""
        if not args_str:
            return Action(tool_name=tool_name, tool_args={})

        try:
            parsed_args = json.loads(args_str)
            if isinstance(parsed_args, dict):
                return Action(tool_name=tool_name, tool_args=parsed_args)
            return Action(tool_name=tool_name, tool_args={"input": parsed_args})
        except json.JSONDecodeError:
            # If standard JSON fails, handle common single-quote replacement safely
            sanitized = args_str.replace("'", '"')
            try:
                parsed_args = json.loads(sanitized)
                if isinstance(parsed_args, dict):
                    return Action(tool_name=tool_name, tool_args=parsed_args)
            except Exception:
                pass
            # Fallback: store raw text so the tool execution or error-handler can see it
            return Action(tool_name=tool_name, tool_args={"_raw_args": args_str})


# ============================================================================
# 4. Pure Python Agent Loop (Think -> Act -> Look)
# ============================================================================

class AgentLoop:
    """
    Executes the autonomous Think -> Act -> Look loop.
    
    Attributes:
        tools: Dictionary mapping tool names to callable Python functions
               or a tool registry object with an .execute(name, args) method.
        llm_caller: A function taking a string prompt and returning a string response.
        max_steps: Maximum loop iterations to protect against infinite loops.
        verbose: When True, prints each step to the console (great for demos).
    """

    def __init__(
        self,
        tools: Dict[str, Callable[..., Any]] | Any,
        llm_caller: Callable[[str], str],
        memory: Optional[Any] = None,
        enable_planning: bool = False,
        safety_gate: Optional[Any] = None,
        audit_logger: Optional[Any] = None,
        max_steps: int = 10,
        verbose: bool = True,
        **kwargs: Any,
    ):
        self.tools = tools
        self.llm_caller = llm_caller
        self.memory = memory if memory is not None else kwargs.get("memory_store")
        self.enable_planning = enable_planning or kwargs.get("enable_planner", False)
        self.safety_gate = safety_gate
        self.audit_logger = audit_logger
        self.max_steps = max_steps
        self.verbose = verbose
        self.trajectory: List[StepRecord] = []
        self.plan: Optional[Any] = None

    def _get_tool_descriptions(self) -> str:
        """Generates tool documentation for the system prompt."""
        if hasattr(self.tools, "get_descriptions"):
            return self.tools.get_descriptions()
        if isinstance(self.tools, dict):
            lines = []
            for name, func in self.tools.items():
                doc = (func.__doc__ or "No description provided.").strip()
                lines.append(f"- {name}: {doc}")
            return "\n".join(lines)
        return "No tools available."

    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Observation:
        """
        Dispatches the tool call to the matching Python function.
        Catches runtime exceptions and turns them into informative, structured
        error observations for self-healing (Part 3).
        Checks safety gate for sensitive tools before execution (Part 6).
        """
        # 1. Safety Gate Intercept (Part 6: "Asks You First")
        if self.safety_gate and hasattr(self.tools, "get"):
            tool_obj = self.tools.get(tool_name)
            if tool_obj and getattr(tool_obj, "is_sensitive", False):
                if self.verbose:
                    print(f"\n🛑 [SAFETY GATE TRIGGERED] Intercepting sensitive tool: {tool_name}")
                approved = self.safety_gate.request_approval(tool_name, tool_args)
                if not approved:
                    if self.verbose:
                        print(f"❌ [ACTION DENIED] Human denied execution of {tool_name}.")
                    denial_payload = {
                        "status": "denied",
                        "tool": tool_name,
                        "message": f"Action '{tool_name}' was DENIED by the user. You must NOT execute this action.",
                        "suggestion": "Acknowledge the user's decision and revise your plan or answer.",
                    }
                    if self.audit_logger:
                        self.audit_logger.log("ACTION_DENIED", {"tool": tool_name, "args": tool_args})
                    return Observation(content=json.dumps(denial_payload), is_error=True)
                else:
                    if self.verbose:
                        print(f"✅ [ACTION AUTHORIZED] Human approved execution of {tool_name}.")

        if self.audit_logger:
            self.audit_logger.log("ACTION_EXECUTE", {"tool": tool_name, "args": tool_args})

        try:
            # Check if using a ToolRegistry object
            if hasattr(self.tools, "execute"):
                result = self.tools.execute(tool_name, tool_args)
                result_str = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                if self.audit_logger:
                    self.audit_logger.log("OBSERVATION", {"tool": tool_name, "result": result, "is_error": False})
                return Observation(content=result_str, is_error=False)

            # Check if using a plain dictionary of functions
            if isinstance(self.tools, dict):
                if tool_name not in self.tools:
                    available = ", ".join(self.tools.keys())
                    error_payload = {
                        "status": "error",
                        "error_type": "ToolNotFoundError",
                        "tool_attempted": tool_name,
                        "message": f"Tool '{tool_name}' does not exist.",
                        "suggestion": f"Choose from available tools: {available}",
                    }
                    if self.audit_logger:
                        self.audit_logger.log("OBSERVATION", {"tool": tool_name, "error": error_payload, "is_error": True})
                    return Observation(content=json.dumps(error_payload), is_error=True)

                tool_func = self.tools[tool_name]
                result = tool_func(**tool_args)
                result_str = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                if self.audit_logger:
                    self.audit_logger.log("OBSERVATION", {"tool": tool_name, "result": result, "is_error": False})
                return Observation(content=result_str, is_error=False)

            error_payload = {
                "status": "error",
                "error_type": "ConfigurationError",
                "message": "Tools registry is not configured correctly.",
            }
            return Observation(content=json.dumps(error_payload), is_error=True)

        except Exception as exc:
            # Transform Python exceptions into structured self-healing observations
            error_payload = {
                "status": "error",
                "error_type": type(exc).__name__,
                "tool_attempted": tool_name,
                "message": str(exc),
                "suggestion": f"Review error message, adjust parameters for '{tool_name}', and retry.",
            }
            if self.audit_logger:
                self.audit_logger.log("OBSERVATION", {"tool": tool_name, "error": error_payload, "is_error": True})
            return Observation(content=json.dumps(error_payload), is_error=True)

    def run(self, user_goal: str) -> AgentResponse:
        """
        Executes the ReAct loop until completion, final answer, or max_steps.
        
        Args:
            user_goal: The user's input request or question.
            
        Returns:
            AgentResponse containing the final answer, trajectory, and status.
        """
        self.trajectory.clear()
        tool_docs = self._get_tool_descriptions()
        system_prompt = REACT_SYSTEM_PROMPT.format(tool_descriptions=tool_docs)

        # Inject persistent memory context if available (Part 4)
        if self.memory and hasattr(self.memory, "retrieve_context"):
            mem_context = self.memory.retrieve_context(user_goal)
            if mem_context:
                system_prompt = f"{system_prompt}\n{mem_context}"
                if self.verbose:
                    print("💾 [MEMORY LOADED] Injected relevant context from memory_store.json")

        # 1. Pre-Action Planning Phase (Part 5)
        self.plan = None
        if self.enable_planning:
            try:
                from .planner import PlannerEngine
            except (ImportError, ValueError):
                from planner import PlannerEngine
            planner = PlannerEngine(self.llm_caller)
            self.plan = planner.create_plan(user_goal, tool_docs)
            if self.verbose:
                print("\n📋 [PRE-ACTION EXECUTION PLAN ESTABLISHED]")
                print(self.plan.to_prompt_text())

        # The scratchpad holds the running dialogue: Thought -> Action -> Observation
        scratchpad = f"Question: {user_goal}\n"

        if self.verbose:
            print("=" * 70)
            print("🤖 STARTING AGENT REACT LOOP (Think -> Act -> Look)")
            print(f"Goal: {user_goal}")
            print("=" * 70)

        if self.audit_logger:
            self.audit_logger.log("TASK_START", {"goal": user_goal})

        step = 0
        while step < self.max_steps:
            step += 1
            plan_prompt = f"\n<ExecutionPlan>\n{self.plan.to_prompt_text()}\n</ExecutionPlan>\n" if self.plan else ""
            current_prompt = f"{system_prompt}{plan_prompt}\n{scratchpad}"

            if self.verbose:
                print(f"\n--- [Turn {step}/{self.max_steps}] ---")

            # 1. THINK & ACT: Call LLM
            llm_output = self.llm_caller(current_prompt).strip()

            # 2. PARSE OUTPUT
            thought = ReActParser.parse_thought(llm_output)
            final_answer = ReActParser.parse_final_answer(llm_output)
            action = ReActParser.parse_action(llm_output)

            record = StepRecord(step_number=step, thought=thought, action=action)

            if thought and self.audit_logger:
                self.audit_logger.log("THOUGHT", {"step": step, "thought": thought.content})

            if self.verbose and thought:
                print(f"🧠 {thought}")

            # Check if agent arrived at Final Answer
            if final_answer is not None:
                if self.verbose:
                    print(f"\n🎯 Final Answer:\n{final_answer}")
                self.trajectory.append(record)

                if self.audit_logger:
                    self.audit_logger.log("TASK_COMPLETED", {"step": step, "final_answer": final_answer})

                # Mark all remaining plan steps completed if reached final answer
                if self.plan:
                    for s in self.plan.steps:
                        s.status = "COMPLETED"

                # Persist task completion to memory store (Part 4)
                if self.memory and hasattr(self.memory, "save_task_summary"):
                    self.memory.save_task_summary(
                        task_id=f"task_{int(datetime.now().timestamp())}",
                        goal=user_goal,
                        outcome=final_answer,
                    )

                return AgentResponse(
                    final_answer=final_answer,
                    success=True,
                    total_steps=step,
                    trajectory=self.trajectory,
                    stop_reason="completed",
                    plan=self.plan,
                )

            # If no action and no final answer, ask model to clarify or treat output as answer
            if action is None:
                # If neither Action nor Final Answer was parsed, treat whole response as answer
                fallback_answer = llm_output
                if self.verbose:
                    print(f"⚠️ No Action detected. Treating response as Final Answer:\n{fallback_answer}")
                self.trajectory.append(record)
                return AgentResponse(
                    final_answer=fallback_answer,
                    success=True,
                    total_steps=step,
                    trajectory=self.trajectory,
                    stop_reason="completed",
                    plan=self.plan,
                )

            # 3. ACT: Print and prepare tool invocation
            if self.verbose:
                print(f"⚡ {action}")

            # 4. LOOK: Execute tool and obtain Observation
            observation = self._execute_tool(action.tool_name, action.tool_args)
            record.observation = observation
            self.trajectory.append(record)

            if self.verbose:
                print(f"👁️  {observation}")

            # Update plan progress (Part 5)
            if self.plan:
                if not observation.is_error:
                    self.plan.advance_next()
                    if self.verbose:
                        print("📋 [PLAN UPDATED] Step advanced to COMPLETED.")
                else:
                    if self.verbose:
                        print("📋 [PLAN NOTE] Step encountered error; keeping active for self-healing.")

            # 5. UPDATE SCRATCHPAD: Feed observation into the prompt for the next turn
            scratchpad += (
                f"{thought}\n" if thought else ""
            ) + f"{action}\n{observation}\n"

        # If loop finishes without reaching Final Answer
        max_step_msg = f"Agent stopped: maximum step limit ({self.max_steps}) reached."
        if self.verbose:
            print(f"\n🛑 {max_step_msg}")

        if self.audit_logger:
            self.audit_logger.log("STEP_LIMIT_EXCEEDED", {"step": step, "max_steps": self.max_steps, "message": max_step_msg})

        return AgentResponse(
            final_answer=max_step_msg,
            success=False,
            total_steps=step,
            trajectory=self.trajectory,
            stop_reason="max_steps_exceeded",
            plan=self.plan,
        )

