#!/usr/bin/env python3
"""
Example: 3 DeepAgents collaborating through a shared filesystem.

Workspace: multi_agent/agent_workspace/ - everything stays here.

Mode 1 (default): phased parallel team workflow.
Mode 2 (--peer): 3 independent agents with versioned task files.

Run:
  cd multi_agent && python multi_agent_example.py "Your task"
  cd multi_agent && python multi_agent_example.py --peer "Your task"
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from langchain.agents.middleware import AgentMiddleware

# Workspace — рядом со скриптом, всё в одной директории
WORKSPACE = Path(__file__).parent / "agent_workspace"
HISTORY_FILE = WORKSPACE / "history.log"
ERROR_FILE = WORKSPACE / "error.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("multi_agent")
logging.getLogger("httpx").setLevel(logging.WARNING)
KNOWN_AGENT_NAMES = frozenset({"main", "researcher", "coder", "reviewer"})
INVALID_EXECUTE_STREAK_LIMIT = 5
INVALID_TOOL_ERROR_STREAK_LIMIT = 8
WORKER_CONTEXT_RETRY_LIMIT = 3
WORKER_TIMEOUT_SECONDS = 720


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content or "")


def _is_context_overflow_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = (
        "context length",
        "input (",
        "longer than the model's context length",
        "contextoverflowerror",
        "too many tokens",
    )
    return any(marker in text for marker in markers)


def _is_graph_recursion_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return exc.__class__.__name__ == "GraphRecursionError" or "recursion limit" in text


def _resolve_agent_name(namespace: object) -> str:
    if not namespace:
        return "main"
    if isinstance(namespace, tuple):
        for part in reversed(namespace):
            if isinstance(part, str) and part in KNOWN_AGENT_NAMES:
                return part
        first = namespace[0]
    else:
        first = namespace

    if isinstance(first, str):
        if first in KNOWN_AGENT_NAMES:
            return first
        if first == "tools" or first.startswith("tools:") or len(first) > 20:
            return "main"
        return first
    return "main"


def build_model():
    from langchain_openai import ChatOpenAI

    # Lower temperature improves tool-calling reliability for local models.
    model = ChatOpenAI(
        model="qwen",
        base_url="http://localhost:30000/v1",
        api_key="not-needed",
        temperature=0.0,
        max_tokens=2048,
    )
    # Force DeepAgents' built-in summarizer to compact much earlier than the
    # model's hard limit. This avoids reaching the 131k-token ceiling first.
    model.profile["max_input_tokens"] = 60000
    return model


def augment_subagent_prompt(prompt: str) -> str:
    return prompt + """

Execution rules:
- Keep tool outputs small and targeted. Prefer short commands and concise flags.
- Never run package managers or privileged commands such as apt, sudo, or system-wide installers.
- Do not paste large command output into the chat. Store details in files and summarize the outcome briefly.
- Trust files more than chat history. Persist important conclusions to board.md or the role-specific report files.
- Always use workspace-relative paths such as `board.md`, `research/report.md`, or `code/file.py`.
- Never use leading-slash pseudo-absolute paths like `/board.md` or `/research/spec.md`.
- If a target file already exists, prefer `read_file` + `edit_file` over `write_file`.
- If a shell command fails, diagnose it and retry with a simpler command instead of spamming variants.
- For `execute`, always pass a JSON object with a non-empty `command` field.
- Valid examples: execute({"command":"rg -n \"TODO\" code/"}) and execute({"command":"curl -I https://example.com"})
- Invalid examples: execute({}), execute({"timeout":"30"}), execute({"commandcurl -s https://example.com | head":""})
- For `write_file`, always pass {"file_path": "relative/path.ext", "content": "..."} where both fields are plain strings.
- Never pass a dict or list as the `content` field of write_file — always serialize to a string first.
- If the file content is large (>200 lines), write it in sections using edit_file after an initial write_file.
"""


def _console_preview(text: str, max_chars: int = 180) -> str:
    compact = text.replace("\n", " ").strip()
    if len(compact) > max_chars:
        return compact[:max_chars] + "..."
    return compact


def _tool_args_to_text(args: object) -> str:
    if isinstance(args, str):
        return args
    try:
        return json.dumps(args or {}, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(args)


def _normalize_tool_args(args: object) -> object:
    if not isinstance(args, str):
        return args or {}
    stripped = args.strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
    except Exception:
        return {"_raw_args": stripped}
    if isinstance(parsed, dict):
        return parsed
    return {"_value": parsed}


def _normalize_workspace_tool_path(path_value: object) -> str | None:
    if not isinstance(path_value, str):
        return None
    normalized = path_value.strip()
    if not normalized:
        return None
    if normalized.startswith("/"):
        normalized = normalized.lstrip("/")
    return normalized or None


def _coerce_text_arg(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


def _has_required_tool_args(tool_name: str, tool_args: object) -> bool:
    if not isinstance(tool_args, dict):
        return False
    if tool_name == "execute":
        return bool(tool_args.get("command"))
    if tool_name == "task":
        return bool(tool_args.get("description"))
    if tool_name == "write_todos":
        return bool(tool_args.get("todos"))
    return True


def _is_tool_error(tool_name: str, content: str) -> bool:
    stripped = content.strip()
    lowered = stripped.lower()

    if tool_name in {"read_file", "glob", "grep", "ls"}:
        return lowered.startswith("error:")

    if tool_name in {"write_file", "edit_file"}:
        return lowered.startswith("error:")

    if tool_name == "write_todos":
        return lowered.startswith("error:")

    if tool_name == "task":
        markers = (
            "error:",
            "we cannot invoke subagent",
            "tool call id is required",
        )
        return any(marker in lowered for marker in markers)

    if tool_name == "execute":
        markers = (
            "error invoking tool",
            "[command failed with exit code",
            "permission denied",
            "syntax error",
            "field required",
            "not found",
            "cannot create ",
            "failed to ",
        )
        return any(marker in lowered for marker in markers)

    return lowered.startswith("error:")


def _is_tool_validation_error(tool_name: str, content: str) -> bool:
    if tool_name not in {"write_file", "edit_file", "read_file", "write_todos", "ls", "task"}:
        return False
    lowered = content.lower()
    markers = (
        "field required",
        "requires a non-empty",
        "requires a string",
        "requires object args",
        "tool call id is required",
    )
    return any(marker in lowered for marker in markers)


def log_error_event(agent: str, tool_name: str, tool_args: object, error_text: str) -> None:
    timestamp = datetime.now(UTC).strftime("%H:%M:%S")
    lines = [
        f"[{timestamp}] [{agent}]",
        f"tool: {tool_name}",
        f"args: {_tool_args_to_text(tool_args)}",
        "error:",
        error_text.rstrip(),
        "",
    ]
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    with ERROR_FILE.open("a", encoding="utf-8") as error_log:
        error_log.write("\n".join(lines))


def log_history(msg: str, agent: str = "system", console_msg: str | None = None) -> None:
    line = f"[{datetime.now(UTC).strftime('%H:%M:%S')}] [{agent}] {msg}\n"
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    with HISTORY_FILE.open("a", encoding="utf-8") as history:
        history.write(line)
    logger.info("[%s] %s", agent, console_msg if console_msg is not None else msg)


def _ensure_tool_call_id(tool_call: dict) -> str:
    """Return the existing tool_call id, generating and injecting a fallback if absent."""
    tc_id = tool_call.get("id")
    if not tc_id:
        tc_id = f"tc_{datetime.now(UTC).strftime('%H%M%S%f')}"
        tool_call["id"] = tc_id
    return str(tc_id)


class ExecutePreflightMiddleware(AgentMiddleware):
    """Normalize or reject malformed execute calls before shell execution."""

    name = "execute_preflight"

    @staticmethod
    def _prepare_args(raw_args: object) -> tuple[dict[str, object] | None, str | None]:
        args = _normalize_tool_args(raw_args)
        if not isinstance(args, dict):
            return None, f"Error: execute requires object args, got {type(args).__name__}."

        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return None, (
                "Error: execute requires a non-empty 'command' string. "
                f"Received args: {_tool_args_to_text(args)}"
            )

        normalized = command.replace("\r\n", "\n").replace("\r", "\n").strip()
        normalized = re.sub(r"\s*\n+\s*", " ", normalized)
        normalized = re.sub(r"2>&(?=\s*\|)", "2>&1", normalized)
        normalized = re.sub(r"2>&\s*$", "2>&1", normalized)

        lowered = normalized.lower()
        if "<parameter" in lowered or "</parameter" in lowered:
            return None, "Error: execute command contains malformed XML-style parameter text."

        patched = dict(args)
        patched["command"] = normalized
        return patched, None

    def wrap_tool_call(self, request, handler):
        from langchain_core.messages import ToolMessage

        tool_call = request.tool_call
        if tool_call.get("name") != "execute":
            _ensure_tool_call_id(tool_call)
            return handler(request)

        patched_args, error = self._prepare_args(tool_call.get("args", {}))
        if error is not None:
            tool_call_id = _ensure_tool_call_id(tool_call)
            return ToolMessage(
                content=error,
                name="execute",
                tool_call_id=tool_call_id,
            )

        tool_call["args"] = patched_args
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        from langchain_core.messages import ToolMessage

        tool_call = request.tool_call
        if tool_call.get("name") != "execute":
            _ensure_tool_call_id(tool_call)
            return await handler(request)

        patched_args, error = self._prepare_args(tool_call.get("args", {}))
        if error is not None:
            tool_call_id = _ensure_tool_call_id(tool_call)
            return ToolMessage(
                content=error,
                name="execute",
                tool_call_id=tool_call_id,
            )

        tool_call["args"] = patched_args
        return await handler(request)


class FilesystemPreflightMiddleware(AgentMiddleware):
    """Normalize filesystem paths and make repeated writes idempotent."""

    name = "filesystem_preflight"

    @staticmethod
    def _prepare_args(
        tool_name: str,
        raw_args: object,
    ) -> tuple[str | None, dict[str, object] | None, str | None]:
        args = _normalize_tool_args(raw_args)
        if not isinstance(args, dict):
            return None, None, f"Error: {tool_name} requires object args, got {type(args).__name__}."

        patched = dict(args)

        if tool_name in {"read_file", "write_file", "edit_file"}:
            file_path = _normalize_workspace_tool_path(
                patched.get("file_path")
                or patched.get("path")
                or patched.get("filename")
                or patched.get("file")
            )
            if not file_path:
                return None, None, f"Error: {tool_name} requires a non-empty 'file_path' string."
            patched["file_path"] = file_path

            if tool_name == "write_file":
                content = _coerce_text_arg(
                    patched.get("content")
                    if "content" in patched
                    else patched.get("contents")
                    or patched.get("text")
                    or patched.get("new_string")
                    or patched.get("value")
                )
                if content is None:
                    return None, None, "Error: write_file requires a string 'content' field."
                patched["content"] = content

                abs_path = WORKSPACE / file_path
                if abs_path.exists() and abs_path.is_file():
                    existing_text = abs_path.read_text(encoding="utf-8")
                    return (
                        "edit_file",
                        {
                            "file_path": file_path,
                            "old_string": existing_text,
                            "new_string": content,
                        },
                        None,
                    )

            if tool_name == "edit_file":
                old_string = _coerce_text_arg(
                    patched.get("old_string")
                    if "old_string" in patched
                    else patched.get("oldText")
                    or patched.get("old_text")
                )
                new_string = _coerce_text_arg(
                    patched.get("new_string")
                    if "new_string" in patched
                    else patched.get("newText")
                    or patched.get("new_text")
                    or patched.get("content")
                    or patched.get("text")
                )
                if old_string is not None:
                    patched["old_string"] = old_string
                if new_string is not None:
                    patched["new_string"] = new_string

            return tool_name, patched, None

        if tool_name == "ls":
            path = _normalize_workspace_tool_path(patched.get("path"))
            patched["path"] = path or "."
            return tool_name, patched, None

        return tool_name, patched, None

    def wrap_tool_call(self, request, handler):
        from langchain_core.messages import ToolMessage

        tool_call = request.tool_call
        tool_name = tool_call.get("name")
        if tool_name not in {"read_file", "write_file", "edit_file", "ls"}:
            _ensure_tool_call_id(tool_call)
            return handler(request)

        patched_name, patched_args, error = self._prepare_args(tool_name, tool_call.get("args", {}))
        if error is not None:
            tool_call_id = _ensure_tool_call_id(tool_call)
            return ToolMessage(
                content=error,
                name=str(tool_name or "filesystem"),
                tool_call_id=tool_call_id,
            )

        tool_call["name"] = patched_name
        tool_call["args"] = patched_args
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        from langchain_core.messages import ToolMessage

        tool_call = request.tool_call
        tool_name = tool_call.get("name")
        if tool_name not in {"read_file", "write_file", "edit_file", "ls"}:
            _ensure_tool_call_id(tool_call)
            return await handler(request)

        patched_name, patched_args, error = self._prepare_args(tool_name, tool_call.get("args", {}))
        if error is not None:
            tool_call_id = _ensure_tool_call_id(tool_call)
            return ToolMessage(
                content=error,
                name=str(tool_name or "filesystem"),
                tool_call_id=tool_call_id,
            )

        tool_call["name"] = patched_name
        tool_call["args"] = patched_args
        return await handler(request)


# --- Subagents ---
RESEARCHER = {
    "name": "researcher",
    "description": (
        "Researches the problem, gathers context, and writes concise reports. "
        "Store results in research/ and board.md. Use write_todos for planning, plus read_file, write_file, and execute."
    ),
    "system_prompt": """You are Researcher, the research specialist.

For complex tasks (3+ steps), use write_todos: create a plan, mark a step in_progress before starting it, and mark it completed immediately after finishing it. Do not batch completions.

Workflow:
- Write reports to research/report.md
- Update board.md in the [Researcher] section
- Use execute for rg, curl, and lightweight shell inspection
- Keep outputs concise, structured, and decision-oriented""",
}

ANALYST = {
    "name": "analyst",
    "description": (
        "Turns the task into acceptance criteria, edge cases, and a concrete implementation spec. "
        "Store results in research/spec.md or review/acceptance.md."
    ),
    "system_prompt": """You are Analyst, the specification specialist.

For complex tasks (3+ steps), use write_todos: create a plan, mark a step in_progress before starting it, and mark it completed immediately after finishing it.

Workflow:
- Extract acceptance criteria, constraints, and edge cases
- Write a precise implementation spec to research/spec.md when planning
- Write an acceptance verification report to review/acceptance.md when validating
- Use only file tools and write_todos; never call execute
- Keep outputs concise, concrete, and testable""",
}

CODER = {
    "name": "coder",
    "description": (
        "Implements solutions from the specification. Read research/ and board.md, then build the solution in code/. "
        "Use write_todos for planning, plus read_file, write_file, edit_file, and execute."
    ),
    "system_prompt": """You are Coder, the implementation specialist.

For complex tasks, use write_todos: plan -> in_progress -> completed as you work. Mark each step completed immediately after you finish it.

Workflow:
- Read research/report.md and board.md
- Write code in code/
- Update board.md in the [Coder] section
- Use execute for tests and linters""",
}

REVIEWER = {
    "name": "reviewer",
    "description": (
        "Reviews code and reports for bugs, regressions, and missing tests. Read research/, code/, and board.md. "
        "Write the review to review/. Use write_todos for the review checklist."
    ),
    "system_prompt": """You are Reviewer, the quality specialist.

For reviews, use write_todos as a checklist for style, logic, tests, and risks. Mark items completed as you verify them.

Workflow:
- Read research/report.md, code/, and board.md
- Write the review to review/review.md
- Update board.md in the [Reviewer] section
- Prioritize findings, risks, and concrete recommendations""",
}


def build_agent():
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend, LocalShellBackend
    from deepagents.graph import BASE_AGENT_PROMPT
    from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
    from deepagents.middleware.subagents import SubAgentMiddleware
    from deepagents.middleware.summarization import create_summarization_middleware
    from langchain.agents import create_agent
    from langchain.agents.middleware import TodoListMiddleware
    from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
    from langgraph.checkpoint.memory import MemorySaver

    model = build_model()
    main_backend = FilesystemBackend(root_dir=WORKSPACE, virtual_mode=True)
    shell_backend = LocalShellBackend(root_dir=WORKSPACE, inherit_env=True, virtual_mode=True)

    def build_compiled_subagent(spec: dict[str, str]):
        runnable = create_deep_agent(
            model=model,
            system_prompt=augment_subagent_prompt(spec["system_prompt"]),
            backend=shell_backend,
            interrupt_on={},
            checkpointer=MemorySaver(),
            name=spec["name"],
        ).with_config({"recursion_limit": 150})
        return {
            "name": spec["name"],
            "description": spec["description"],
            "runnable": runnable,
        }

    compiled_subagents = [
        build_compiled_subagent(RESEARCHER),
        build_compiled_subagent(CODER),
        build_compiled_subagent(REVIEWER),
    ]

    main_system_prompt = """You are the team orchestrator for researcher, coder, and reviewer.

You are a coordinator, not an implementer.
Always use the full delegation flow, even for simple tasks:
1. researcher researches
2. coder implements
3. reviewer reviews

Start with write_todos for that plan and mark items in_progress/completed as you delegate.

Delegate only through the task tool using:
- subagent_type="researcher" with a detailed description
- subagent_type="coder" with a detailed description
- subagent_type="reviewer" with a detailed description

Shared coordination happens through files: board.md, research/, code/, review/, manifest.json.
Never perform shell work or direct implementation yourself.
You only have planning and delegation tools. Delegate, then synthesize concise progress updates."""

    agent = create_agent(
        model,
        system_prompt=main_system_prompt + "\n\n" + BASE_AGENT_PROMPT,
        middleware=[
            TodoListMiddleware(),
            SubAgentMiddleware(backend=main_backend, subagents=compiled_subagents),
            create_summarization_middleware(model, main_backend),
            AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
            PatchToolCallsMiddleware(),
        ],
        checkpointer=MemorySaver(),
        name="main",
    ).with_config({"recursion_limit": 200})

    return agent


async def run_with_history(agent, task: str):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    log_history(f"Старт задачи: {task[:80]}...", "system")
    log_history("---", "system")
    current_task = task
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        run_id = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        config = {"configurable": {"thread_id": f"multi-agent-{run_id}-a{attempt}"}}
        input_state = {"messages": [HumanMessage(content=current_task)]}
        seen_tool_calls: set[tuple[str, int | str]] = set()
        ai_buffer: dict[str, str] = {}
        last_tool_call: dict[str, tuple[str, object]] = {}
        tool_calls_by_id: dict[str, tuple[str, object]] = {}

        def _flush_ai(an: str) -> None:
            text = ai_buffer.get(an, "")
            cleaned = text.strip()
            if cleaned:
                log_history(f"AI: {cleaned}", an)
            ai_buffer[an] = ""

        try:
            async for chunk in agent.astream(
                input_state,
                stream_mode=["messages"],
                subgraphs=True,
                config=config,
            ):
                if not isinstance(chunk, tuple) or len(chunk) != 3:
                    continue
                namespace, stream_mode, data = chunk
                agent_name = _resolve_agent_name(namespace)

                if stream_mode == "messages" and isinstance(data, tuple):
                    msg, meta = data[0], data[1] if len(data) > 1 else {}
                    if isinstance(msg, HumanMessage):
                        _flush_ai(agent_name)
                        content = _content_to_text(getattr(msg, "content", ""))
                        if content:
                            log_history(f"User: {content[:200]}...", agent_name)
                    elif isinstance(msg, AIMessage):
                        content = _content_to_text(getattr(msg, "content", ""))
                        if content.strip():
                            ai_buffer[agent_name] = ai_buffer.get(agent_name, "") + content
                        tool_calls = getattr(msg, "tool_calls", []) or []
                        if tool_calls:
                            _flush_ai(agent_name)
                        for i, tc in enumerate(tool_calls):
                            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                            if not name:
                                continue
                            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                            key = (agent_name, tc_id if tc_id is not None else i)
                            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                            args = _normalize_tool_args(args)
                            last_tool_call[agent_name] = (name, args)
                            if tc_id is not None:
                                tool_calls_by_id[str(tc_id)] = (name, args)
                            if key in seen_tool_calls:
                                continue
                            seen_tool_calls.add(key)
                            if name == "execute" and not _has_required_tool_args(name, args):
                                log_history(f"Tool: execute(<invalid args: {args}>)", agent_name)
                                continue
                            if name == "task" and not _has_required_tool_args(name, args):
                                log_history(f"Tool: task(<invalid args: {args}>)", agent_name)
                                continue
                            if name == "write_todos":
                                todos = (args or {}).get("todos", [])
                                if not todos:
                                    continue
                                lines = [f"  - [{t.get('status','?')}] {t.get('content','')}" for t in todos]
                                log_history(f"write_todos:\n" + "\n".join(lines), agent_name)
                            else:
                                if args:
                                    if name == "execute":
                                        cmd = (args or {}).get("command", "")[:80]
                                        log_history(f"Tool: execute({cmd}...)", agent_name)
                                    elif name == "task":
                                        subagent_type = (args or {}).get("subagent_type", "?")
                                        task_preview = ((args or {}).get("description") or "")[:120]
                                        log_history(f"Tool: task({subagent_type}: {task_preview}...)", agent_name)
                                    else:
                                        log_history(f"Tool: {name}({args})", agent_name)
                                else:
                                    log_history(f"Tool: {name}", agent_name)
                    elif isinstance(msg, ToolMessage):
                        _flush_ai(agent_name)
                        content = _content_to_text(getattr(msg, "content", ""))
                        if content:
                            tool_call_id = getattr(msg, "tool_call_id", None)
                            tool_name, tool_args = (
                                tool_calls_by_id.get(str(tool_call_id))
                                if tool_call_id is not None
                                else None
                            ) or last_tool_call.get(agent_name, ("<unknown>", {}))
                            if _is_tool_error(tool_name, content):
                                log_error_event(agent_name, tool_name, tool_args, content)
                            log_history(
                                f"Tool result: {content}",
                                agent_name,
                                console_msg=f"Tool result: {_console_preview(content)}",
                            )

            for an in list(ai_buffer):
                _flush_ai(an)
            log_history("--- Задача завершена", "system")
            return
        except Exception as exc:
            for an in list(ai_buffer):
                _flush_ai(an)
            if attempt >= max_attempts or not _is_context_overflow_error(exc):
                raise
            log_history(
                "Context window exhausted. Restarting with fresh chat state and workspace files as memory.",
                "system",
            )
            current_task = (
                "Continue the existing task from the current workspace state only.\n\n"
                f"Original user task: {task}\n\n"
                "First delegate to researcher to inspect board.md, manifest.json, research/, code/, review/, and tasks/, "
                "then summarize the current status back to you. Treat workspace files as the source of truth and do not rely "
                "on previous chat history. Continue the workflow from there."
            )


def _init_manifest() -> dict:
    import json
    manifest = {"tasks": [], "created_at": datetime.now(UTC).isoformat()}
    path = WORKSPACE / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _update_manifest(task_id: str, status: str, owner: str = "", output: str = "") -> None:
    import json
    path = WORKSPACE / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    entry = next((t for t in data["tasks"] if t["id"] == task_id), None)
    if entry:
        entry["status"] = status
        entry["owner"] = owner
        entry["completed_at"] = datetime.now(UTC).isoformat() if status in {"done", "failed"} else None
        if output:
            entry["output"] = output
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_board(section: str) -> None:
    board_path = WORKSPACE / "board.md"
    with board_path.open("a", encoding="utf-8") as board:
        board.write(section.rstrip() + "\n\n")


def _task_slug(task: str) -> str:
    file_match = re.search(r"([A-Za-z0-9_.-]+)\.(py|js|ts|md|txt|json|ya?ml)\b", task)
    if file_match:
        stem = file_match.group(1).lower()
    else:
        words = re.findall(r"[A-Za-z0-9]+", task.lower())[:6]
        stem = "_".join(words) if words else "task"
    stem = re.sub(r"[^a-z0-9_]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem[:48] or "task"


def _build_worker_agent(spec: dict[str, str], model, backend):
    from deepagents.graph import BASE_AGENT_PROMPT
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
    from deepagents.middleware.summarization import create_summarization_middleware
    from langchain.agents import create_agent
    from langchain.agents.middleware import TodoListMiddleware
    from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
    from langgraph.checkpoint.memory import MemorySaver

    middleware = [
        TodoListMiddleware(),
        FilesystemMiddleware(backend=backend),
        FilesystemPreflightMiddleware(),
        ExecutePreflightMiddleware(),
        create_summarization_middleware(model, backend),
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        PatchToolCallsMiddleware(),
    ]

    extra_role_rules = ""
    if spec["name"] == "analyst":
        extra_role_rules = (
            "\nYou do not have shell access."
            "\nNever call execute."
            "\nUse only read_file, write_file, edit_file, glob, grep, ls, and write_todos."
            "\nIf you need external information, document assumptions and produce the best spec you can from workspace files."
        )

    worker_prompt = (
        augment_subagent_prompt(spec["system_prompt"])
        + "\n\nUse write_todos for any plan with 3+ steps. Mark each step completed immediately after it is done."
        + "\nYou are a worker agent, not an orchestrator."
        + "\nYou do not have subagents. Never try to call the task tool."
        + extra_role_rules
    )

    return create_agent(
        model,
        system_prompt=worker_prompt + "\n\n" + BASE_AGENT_PROMPT,
        middleware=middleware,
        checkpointer=MemorySaver(),
        name=spec["name"],
    ).with_config({"recursion_limit": 150})


async def _run_logged_worker_once(
    *,
    name: str,
    agent,
    message: str,
    thread_id: str,
    manifest_task_id: str,
    output_hint: str = "",
) -> str:
    from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

    _update_manifest(manifest_task_id, "in_progress", name)
    final_status = "done"
    ai_buffer = ""
    seen_tool_calls: set[str] = set()
    last_tool_call: tuple[str, object] = ("<none>", {})
    tool_calls_by_id: dict[str, tuple[str, object]] = {}
    partial_tool_calls: dict[str, dict[str, object]] = {}
    invalid_execute_streak = 0
    invalid_tool_error_streak: dict[str, int] = {}
    abort_worker = False

    def flush_ai() -> None:
        nonlocal ai_buffer
        cleaned = ai_buffer.strip()
        if cleaned:
            log_history(f"AI: {cleaned}", name)
        ai_buffer = ""

    def handle_tool_call(tool_name: str, tool_args: object, *, tool_id: object, dedupe_suffix: object) -> None:
        nonlocal last_tool_call, invalid_execute_streak, abort_worker, final_status
        dedupe_key = f"{tool_name}:{tool_id if tool_id is not None else dedupe_suffix}"
        last_tool_call = (tool_name, tool_args)
        if tool_id is not None:
            tool_calls_by_id[str(tool_id)] = (tool_name, tool_args)
        if dedupe_key in seen_tool_calls:
            return
        seen_tool_calls.add(dedupe_key)

        if tool_name == "write_todos":
            invalid_execute_streak = 0
            todos = tool_args.get("todos", []) if isinstance(tool_args, dict) else []
            if todos:
                lines = [f"  - [{t.get('status', '?')}] {t.get('content', '')}" for t in todos]
                log_history("write_todos:\n" + "\n".join(lines), name)
            return

        if tool_name == "execute":
            command = tool_args.get("command", "") if isinstance(tool_args, dict) else ""
            if command:
                invalid_execute_streak = 0
                log_history(f"Tool: execute({command[:80]}...)", name)
                return

            invalid_execute_streak += 1
            log_history(f"Tool: execute(<invalid args: {tool_args}>)", name)
            if invalid_execute_streak >= INVALID_EXECUTE_STREAK_LIMIT:
                final_status = "failed"
                abort_worker = True
                log_error_event(
                    name,
                    "execute",
                    tool_args,
                    f"Aborted after {invalid_execute_streak} consecutive invalid execute calls.",
                )
                log_history(
                    f"Stopping {name}: too many consecutive invalid execute calls.",
                    "system",
                )
            return

        invalid_execute_streak = 0
        if tool_name == "task":
            if not (isinstance(tool_args, dict) and tool_args.get("description")):
                log_history(f"Tool: task(<invalid args: {tool_args}>)", name)
                return
            subagent_type = tool_args.get("subagent_type", "?")
            task_preview = (tool_args.get("description") or "")[:120]
            log_history(f"Tool: task({subagent_type}: {task_preview}...)", name)
            return

        if isinstance(tool_args, dict) and tool_args:
            log_history(f"Tool: {tool_name}({tool_args})", name)
        else:
            log_history(f"Tool: {tool_name}", name)

    log_history(f"Starting {name}", "system")
    async for chunk in agent.astream(
        {"messages": [HumanMessage(content=message)]},
        stream_mode=["messages"],
        config={"configurable": {"thread_id": thread_id}},
    ):
        if not (isinstance(chunk, tuple) and len(chunk) == 2):
            continue
        stream_mode, data = chunk
        if stream_mode != "messages" or not isinstance(data, tuple) or not data:
            continue

        msg_obj = data[0]
        if isinstance(msg_obj, (AIMessage, AIMessageChunk)):
            content_blocks = getattr(msg_obj, "content_blocks", None)
            if not isinstance(content_blocks, list):
                content = _content_to_text(getattr(msg_obj, "content", ""))
                if content.strip():
                    ai_buffer += content
            else:
                for block in content_blocks:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type == "text":
                        text = block.get("text", "")
                        if isinstance(text, str) and text.strip():
                            ai_buffer += text
                        continue
                    if block_type not in {"tool_call", "tool_call_chunk"}:
                        continue

                    flush_ai()

                    chunk_name = block.get("name")
                    chunk_id = block.get("id")
                    chunk_index = block.get("index")
                    buffer_key = chunk_index if chunk_index is not None else chunk_id
                    if buffer_key is None:
                        continue

                    entry = partial_tool_calls.setdefault(
                        str(buffer_key),
                        {"name": None, "id": None, "args": None, "args_parts": []},
                    )
                    if chunk_name:
                        entry["name"] = chunk_name
                    if chunk_id:
                        entry["id"] = chunk_id

                    chunk_args = block.get("args")
                    if isinstance(chunk_args, dict):
                        entry["args"] = chunk_args
                        entry["args_parts"] = []
                    elif isinstance(chunk_args, str):
                        if chunk_args:
                            parts = entry.setdefault("args_parts", [])
                            if isinstance(parts, list):
                                if not parts or chunk_args != parts[-1]:
                                    parts.append(chunk_args)
                                entry["args"] = "".join(parts)
                    elif chunk_args is not None:
                        entry["args"] = chunk_args

                    parsed_name = entry.get("name")
                    if not isinstance(parsed_name, str) or not parsed_name:
                        continue

                    parsed_args = entry.get("args")
                    if isinstance(parsed_args, str):
                        if not parsed_args:
                            continue
                        try:
                            parsed_args = json.loads(parsed_args)
                        except json.JSONDecodeError:
                            if block_type == "tool_call":
                                parsed_args = _normalize_tool_args(parsed_args)
                            else:
                                continue
                    elif parsed_args is None:
                        continue

                    if not isinstance(parsed_args, dict):
                        parsed_args = {"value": parsed_args}

                    tool_id = entry.get("id")
                    handle_tool_call(
                        parsed_name,
                        parsed_args,
                        tool_id=tool_id,
                        dedupe_suffix=buffer_key,
                    )

                    if tool_id is not None or block_type == "tool_call":
                        partial_tool_calls.pop(str(buffer_key), None)
            if abort_worker:
                break
        elif isinstance(msg_obj, ToolMessage):
            flush_ai()
            content = _content_to_text(getattr(msg_obj, "content", ""))
            if content:
                tool_call_id = getattr(msg_obj, "tool_call_id", None)
                tool_name, tool_args = (
                    tool_calls_by_id.get(str(tool_call_id))
                    if tool_call_id is not None
                    else None
                ) or last_tool_call
                if _is_tool_error(tool_name, content):
                    log_error_event(name, tool_name, tool_args, content)
                    if _is_tool_validation_error(tool_name, content):
                        streak = invalid_tool_error_streak.get(tool_name, 0) + 1
                        invalid_tool_error_streak[tool_name] = streak
                        if streak >= INVALID_TOOL_ERROR_STREAK_LIMIT:
                            final_status = "failed"
                            abort_worker = True
                            log_error_event(
                                name,
                                tool_name,
                                tool_args,
                                (
                                    "Aborted after repeated malformed tool calls. "
                                    f"Last error: {content}"
                                ),
                            )
                            log_history(
                                f"Stopping {name}: too many consecutive malformed tool calls for {tool_name}.",
                                "system",
                            )
                else:
                    invalid_tool_error_streak.pop(tool_name, None)
                log_history(
                    f"Tool result: {content}",
                    name,
                    console_msg=f"Tool result: {_console_preview(content)}",
                )
        if abort_worker:
            break

    flush_ai()
    _update_manifest(manifest_task_id, final_status, name, output_hint)
    log_history(f"{name} {final_status}", "system")
    return final_status


async def _run_logged_worker(
    *,
    name: str,
    agent,
    message: str,
    thread_id: str,
    manifest_task_id: str,
    output_hint: str = "",
) -> str:
    current_message = message

    for attempt in range(1, WORKER_CONTEXT_RETRY_LIMIT + 1):
        try:
            return await _run_logged_worker_once(
                name=name,
                agent=agent,
                message=current_message,
                thread_id=f"{thread_id}-a{attempt}",
                manifest_task_id=manifest_task_id,
                output_hint=output_hint,
            )
        except Exception as exc:
            if _is_graph_recursion_error(exc):
                _update_manifest(manifest_task_id, "failed", name, output_hint)
                log_error_event(
                    name,
                    "<worker>",
                    {"thread_id": thread_id, "attempt": attempt},
                    f"Graph recursion limit reached on attempt {attempt}: {exc}",
                )
                log_history(
                    f"{name} failed: recursion limit reached without a stop condition.",
                    "system",
                )
                return "failed"

            if not _is_context_overflow_error(exc):
                raise

            if attempt >= WORKER_CONTEXT_RETRY_LIMIT:
                _update_manifest(manifest_task_id, "failed", name, output_hint)
                log_error_event(
                    name,
                    "<worker>",
                    {"thread_id": thread_id, "attempt": attempt},
                    f"Context overflow after {attempt} attempts: {exc}",
                )
                log_history(
                    f"{name} failed: context overflow persisted after {attempt} attempts.",
                    "system",
                )
                return "failed"

            log_history(
                f"{name} hit context overflow on attempt {attempt}. Restarting with fresh chat state.",
                "system",
            )
            current_message = (
                "Continue the existing worker task from the current workspace state only.\n\n"
                f"Worker role: {name}\n"
                f"Original worker task: {message}\n\n"
                "Treat workspace files as the source of truth. First inspect board.md, manifest.json, research/, "
                "code/, review/, and tasks/ if they exist. Then continue from the latest saved artifacts instead of "
                "relying on prior chat history. Keep the response concise and continue the assigned role only."
            )


async def _run_worker_with_timeout(
    *,
    name: str,
    agent,
    message: str,
    thread_id: str,
    manifest_task_id: str,
    output_hint: str = "",
    timeout_seconds: int | None = WORKER_TIMEOUT_SECONDS,
) -> str:
    worker_coro = _run_logged_worker(
        name=name,
        agent=agent,
        message=message,
        thread_id=thread_id,
        manifest_task_id=manifest_task_id,
        output_hint=output_hint,
    )

    if timeout_seconds is None:
        return await worker_coro

    try:
        return await asyncio.wait_for(
            worker_coro,
            timeout=timeout_seconds,
        )
    except TimeoutError:
        _update_manifest(manifest_task_id, "failed", name, output_hint)
        log_error_event(
            name,
            "<worker>",
            {"thread_id": thread_id, "timeout_seconds": timeout_seconds},
            f"Timed out after {timeout_seconds} seconds without completing.",
        )
        log_history(
            f"{name} failed: timed out after {timeout_seconds} seconds.",
            "system",
        )
        return "failed"


async def run_default_parallel_mode(task: str):
    import json
    import contextlib
    from deepagents.backends import FilesystemBackend, LocalShellBackend

    model = build_model()
    shell_backend = LocalShellBackend(root_dir=WORKSPACE, inherit_env=True, virtual_mode=True)
    fs_backend = FilesystemBackend(root_dir=WORKSPACE, virtual_mode=True)
    task_slug = _task_slug(task)
    report_path = f"research/{task_slug}_report.md"
    spec_path = f"research/{task_slug}_spec.md"
    review_path = f"review/{task_slug}_review.md"
    acceptance_path = f"review/{task_slug}_acceptance.md"

    task_id = datetime.now(UTC).strftime("task_%Y%m%d_%H%M%S")
    manifest_path = WORKSPACE / "manifest.json"

    _init_manifest()
    (WORKSPACE / "board.md").write_text(
        "# Task Board\n\n"
        f"Task: {task}\n\n"
        "## Planned Flow\n"
        "1. Parallel discovery: researcher + analyst\n"
        "2. Implementation: coder\n"
        "3. Parallel validation: reviewer + analyst\n\n"
        "## Artifact Paths\n"
        f"- Report: `{report_path}`\n"
        f"- Spec: `{spec_path}`\n"
        f"- Review: `{review_path}`\n"
        f"- Acceptance: `{acceptance_path}`\n\n",
        encoding="utf-8",
    )

    researcher_agent = _build_worker_agent(RESEARCHER, model, shell_backend)
    analyst_agent = _build_worker_agent(ANALYST, model, fs_backend)
    coder_agent = _build_worker_agent(CODER, model, shell_backend)
    reviewer_agent = _build_worker_agent(REVIEWER, model, fs_backend)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    created_at = datetime.now(UTC).isoformat()
    manifest["tasks"] = [
        {"id": f"{task_id}:researcher", "status": "pending", "owner": "researcher", "created_at": created_at},
        {"id": f"{task_id}:analyst-plan", "status": "pending", "owner": "analyst", "created_at": created_at},
        {"id": f"{task_id}:coder", "status": "pending", "owner": "coder", "created_at": created_at},
        {"id": f"{task_id}:reviewer", "status": "pending", "owner": "reviewer", "created_at": created_at},
        {"id": f"{task_id}:analyst-validate", "status": "pending", "owner": "analyst", "created_at": created_at},
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    log_history(f"Старт задачи: {task[:80]}...", "system")
    log_history("---", "system")

    run_id = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    researcher_task = asyncio.create_task(
        _run_worker_with_timeout(
            name="researcher",
            agent=researcher_agent,
            message=(
                f"Task: {task}\n\n"
                "Read board.md. Research the problem, likely implementation approach, constraints, dependencies, and risks. "
                f"Write the result to {report_path}. Do not edit board.md in this phase."
            ),
            thread_id=f"default-{run_id}-researcher",
            manifest_task_id=f"{task_id}:researcher",
            output_hint=report_path,
            timeout_seconds=None,
        )
    )
    analyst_plan_status = await _run_worker_with_timeout(
        name="analyst",
        agent=analyst_agent,
        message=(
            f"Task: {task}\n\n"
            "Read board.md. Produce acceptance criteria, edge cases, and a concrete implementation spec. "
            f"Write the result to {spec_path}. Do not edit board.md in this phase."
        ),
        thread_id=f"default-{run_id}-analyst-plan",
        manifest_task_id=f"{task_id}:analyst-plan",
        output_hint=spec_path,
    )

    researcher_status = "running"
    if researcher_task.done():
        researcher_status = await researcher_task
    else:
        log_history(
            "researcher still running after phase 1 kickoff; continuing with latest available research artifacts.",
            "system",
        )

    if analyst_plan_status != "done":
        _append_board(
            "## Phase 1 Aborted\n"
            "- Analyst planning worker failed.\n"
            "- Implementation phase was skipped to avoid compounding bad state."
        )
        log_history("Stopping pipeline after failed phase 1.", "system")
        if not researcher_task.done():
            researcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await researcher_task
        return

    _append_board(
        "## Phase 1 Complete\n"
        f"- `{spec_path}` contains acceptance criteria and edge cases.\n"
        f"- Research status: `{researcher_status}`.\n"
        f"- Expected research artifact path: `{report_path}`."
    )

    coder_status = await _run_worker_with_timeout(
        name="coder",
        agent=coder_agent,
        message=(
            f"Task: {task}\n\n"
            f"Read {report_path} and {spec_path}. Implement the requested solution. "
            "If the task names a specific file path, use that path. Otherwise place new artifacts in code/. "
            "Do not edit board.md."
        ),
        thread_id=f"default-{run_id}-coder",
        manifest_task_id=f"{task_id}:coder",
    )

    if coder_status != "done":
        _append_board(
            "## Phase 2 Aborted\n"
            "- Implementation worker failed.\n"
            "- Validation phase was skipped because there is no trustworthy implementation to review."
        )
        log_history("Stopping pipeline after failed coder phase.", "system")
        if not researcher_task.done():
            researcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await researcher_task
        return

    _append_board("## Phase 2 Complete\n- Implementation produced by `coder`.")

    phase3_statuses = await asyncio.gather(
        _run_worker_with_timeout(
            name="reviewer",
            agent=reviewer_agent,
            message=(
                f"Task: {task}\n\n"
                f"Read {report_path}, {spec_path}, and the produced implementation. "
                "Review for bugs, regressions, missing tests, and unclear assumptions. "
                f"Write findings to {review_path}. Do not edit board.md."
            ),
            thread_id=f"default-{run_id}-reviewer",
            manifest_task_id=f"{task_id}:reviewer",
            output_hint=review_path,
        ),
        _run_worker_with_timeout(
            name="analyst",
            agent=analyst_agent,
            message=(
                f"Task: {task}\n\n"
                f"Read {spec_path} and the produced implementation. "
                f"Verify whether the acceptance criteria were met and write the result to {acceptance_path}. "
                "Do not edit board.md."
            ),
            thread_id=f"default-{run_id}-analyst-validate",
            manifest_task_id=f"{task_id}:analyst-validate",
            output_hint=acceptance_path,
        ),
    )

    if any(status != "done" for status in phase3_statuses):
        _append_board(
            "## Phase 3 Incomplete\n"
            "- At least one validation worker failed.\n"
            "- Review artifacts may be partial and should not be trusted blindly."
        )
        log_history("Validation phase completed with failures.", "system")
        if not researcher_task.done():
            researcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await researcher_task
        return

    _append_board(
        "## Phase 3 Complete\n"
        f"- `{review_path}` contains bug and risk findings.\n"
        f"- `{acceptance_path}` contains acceptance verification."
    )
    if not researcher_task.done():
        researcher_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await researcher_task
    log_history("--- Задача завершена", "system")


async def run_peer_mode(task: str):
    import json
    from deepagents.backends import LocalShellBackend
    from langchain_core.messages import HumanMessage

    model = build_model()

    backend = LocalShellBackend(root_dir=WORKSPACE, inherit_env=True, virtual_mode=True)

    tasks_dir = WORKSPACE / "tasks"
    outputs_dir = WORKSPACE / "outputs"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)
    (WORKSPACE / "research").mkdir(exist_ok=True)
    (WORKSPACE / "code").mkdir(exist_ok=True)
    (WORKSPACE / "review").mkdir(exist_ok=True)

    task_id = "task_001"
    pending = tasks_dir / f"{task_id}_pending.md"
    pending.write_text(f"# Task\n\n{task}\n\n", encoding="utf-8")

    _init_manifest()
    manifest_path = WORKSPACE / "manifest.json"

    (WORKSPACE / "board.md").write_text(f"# Task Board\n\n{task}\n\n", encoding="utf-8")

    def make_agent(name: str, prompt: str):
        return _build_worker_agent({"name": name, "system_prompt": prompt}, model, backend)

    agents = [
        ("researcher", make_agent("researcher", RESEARCHER["system_prompt"]),
         f"Read tasks/{task_id}_pending.md and board.md. Research the task, requirements, constraints, and likely pitfalls. Write the report to research/report.md. Add a [Researcher] section to board.md with findings and open questions."),
        ("coder", make_agent("coder", CODER["system_prompt"]),
         f"Read research/report.md and board.md. Implement the solution in code/. Add a [Coder] section to board.md describing what changed, which files were created, and any remaining risks."),
        ("reviewer", make_agent("reviewer", REVIEWER["system_prompt"]),
         f"Read research/, code/, and board.md. Review the result for bugs, regressions, missing tests, and unclear assumptions. Write the review to review/review.md. Add a [Reviewer] section to board.md with findings and recommendations."),
    ]

    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    created_at = datetime.now(UTC).isoformat()
    m["tasks"] = [
        {"id": f"{task_id}:{name}", "status": "pending", "owner": name, "created_at": created_at}
        for name, _, _ in agents
    ]
    manifest_path.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")

    run_id = datetime.now(UTC).strftime("%Y%m%d%H%M%S")

    for name, agent, msg in agents:
        agent_task_id = f"{task_id}:{name}"
        last_tool_call: tuple[str, object] = ("<none>", {})
        tool_calls_by_id: dict[str, tuple[str, object]] = {}
        inprogress = tasks_dir / f"{task_id}_inprogress_{name}.md"
        if pending.exists():
            pending.rename(inprogress)
        elif not inprogress.exists():
            inprogress.write_text((WORKSPACE / "board.md").read_text(encoding="utf-8"), encoding="utf-8")

        log_history(f"Запуск {name} (claim: {inprogress.name})", name)
        config = {"configurable": {"thread_id": f"peer-{name}-{run_id}"}}
        async for chunk in agent.astream(
            {"messages": [HumanMessage(content=msg)]},
            stream_mode=["messages"],
            config=config,
        ):
            if isinstance(chunk, tuple) and len(chunk) == 3:
                _, stream_mode, data = chunk
                if stream_mode == "messages" and isinstance(data, (list, tuple)) and data:
                    msg_obj = data[0] if isinstance(data, (list, tuple)) else data
                    from langchain_core.messages import AIMessage, ToolMessage
                    if isinstance(msg_obj, AIMessage):
                        c = _content_to_text(getattr(msg_obj, "content", ""))
                        if c:
                            log_history(c[:200], name)
                        for tc in getattr(msg_obj, "tool_calls", []) or []:
                            tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                            args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                            last_tool_call = (tc_name or "<unknown>", args)
                            if tc_id is not None and tc_name:
                                tool_calls_by_id[str(tc_id)] = (tc_name, args)
                            if tc_name == "write_todos":
                                todos = args.get("todos", [])
                                lines = [f"  [{t.get('status','?')}] {t.get('content','')}" for t in todos]
                                log_history("write_todos:\n" + "\n".join(lines), name)
                    elif isinstance(msg_obj, ToolMessage):
                        c = _content_to_text(getattr(msg_obj, "content", ""))
                        tool_call_id = getattr(msg_obj, "tool_call_id", None)
                        tool_name, tool_args = (
                            tool_calls_by_id.get(str(tool_call_id))
                            if tool_call_id is not None
                            else None
                        ) or last_tool_call
                        if _is_tool_error(tool_name, c):
                            log_error_event(name, tool_name, tool_args, c)
                        log_history(
                            f"Tool result: {c}",
                            name,
                            console_msg=f"Tool result: {_console_preview(c)}",
                        )

        done = tasks_dir / f"{task_id}_done_{name}.md"
        if inprogress.exists():
            inprogress.rename(done)
        _update_manifest(agent_task_id, "done", name, f"tasks/{done.name}")
        log_history(f"{name} завершил", "system")


def main():
    args = sys.argv[1:]
    peer_mode = "--peer" in args
    if peer_mode:
        args = [a for a in args if a != "--peer"]

    task = (
        "Create a simple Python script hello.py that prints 'Hello from agents!'. "
        "The Researcher should analyze what is needed, the Coder should implement it, and the Reviewer should check it."
    )
    if args:
        task = " ".join(args)

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    ERROR_FILE.touch(exist_ok=True)
    (WORKSPACE / "board.md").write_text("# Task Board\n\n", encoding="utf-8")
    (WORKSPACE / "research").mkdir(exist_ok=True)
    (WORKSPACE / "code").mkdir(exist_ok=True)
    (WORKSPACE / "review").mkdir(exist_ok=True)
    (WORKSPACE / "tasks").mkdir(exist_ok=True)
    (WORKSPACE / "outputs").mkdir(exist_ok=True)

    if peer_mode:
        asyncio.run(run_peer_mode(task))
    else:
        asyncio.run(run_default_parallel_mode(task))

    print(f"\nИстория: {HISTORY_FILE}")
    print(f"Workspace: {WORKSPACE}")


if __name__ == "__main__":
    main()
