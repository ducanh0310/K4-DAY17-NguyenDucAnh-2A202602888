from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: Baseline Agent with within-session thread memory only.

    - No persistent `User.md` memory
    - Forgets long-term facts across new thread IDs
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None
        if not force_offline:
            self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting."""
        if self.force_offline or self.langchain_agent is None:
            return self._reply_offline(user_id, thread_id, message)
        
        # Live path fallback if configured
        try:
            res = self.langchain_agent.invoke(
                {"messages": [("user", message)]},
                config={"configurable": {"thread_id": thread_id}},
            )
            response_text = res["messages"][-1].content
            agent_t = estimate_tokens(response_text)
            prompt_t = estimate_tokens(message)
            return {"response": response_text, "agent_tokens": agent_t, "prompt_tokens": prompt_t}
        except Exception:
            return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()

        session = self.sessions[thread_id]

        # Calculate prompt token load for this turn (all previous thread messages + current message)
        history_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)
        current_user_tokens = estimate_tokens(message)
        prompt_tokens = history_tokens + current_user_tokens

        session.prompt_tokens_processed += prompt_tokens
        session.messages.append({"role": user_id, "content": message})


        # Deterministic reply logic for baseline
        msg_lower = message.lower()
        
        # Look for facts mentioned ONLY in this current thread's messages
        all_thread_text = " ".join([m["content"] for m in session.messages])

        if "tên" in msg_lower or "ai" in msg_lower:
            if "DũngCT Stress" in all_thread_text:
                resp = "Bạn tên là DũngCT Stress."
            elif "DũngCT" in all_thread_text:
                resp = "Bạn tên là DũngCT."
            else:
                resp = "Chào bạn, mình không rõ tên bạn là gì vì đây là phiên mới."
        elif "đồ uống" in msg_lower:
            if "cà phê sữa đá" in all_thread_text:
                resp = "Đồ uống yêu thích của bạn là cà phê sữa đá."
            else:
                resp = "Mình không nhớ đồ uống yêu thích của bạn."
        elif "nghề" in msg_lower or "công việc" in msg_lower:
            if "MLOps engineer" in all_thread_text:
                resp = "Hiện tại bạn làm MLOps engineer."
            elif "backend engineer" in all_thread_text:
                resp = "Bạn làm backend engineer."
            else:
                resp = "Mình không có thông tin về nghề nghiệp của bạn."
        elif "ở đâu" in msg_lower or "nơi ở" in msg_lower:
            if "Đà Nẵng" in all_thread_text:
                resp = "Bạn đang ở Đà Nẵng."
            elif "Huế" in all_thread_text:
                resp = "Bạn đang ở Huế."
            else:
                resp = "Mình không biết hiện tại bạn đang ở đâu."
        else:
            resp = f"Tôi đã ghi nhận thông tin: '{message[:50]}...' trong thread này."

        agent_tokens = estimate_tokens(resp)
        session.token_usage += agent_tokens
        session.messages.append({"role": "assistant", "content": resp})

        return {
            "response": resp,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        try:
            import importlib
            memory_mod = importlib.import_module("langgraph.checkpoint.memory")
            MemorySaver = getattr(memory_mod, "MemorySaver")
            prebuilt_mod = importlib.import_module("langgraph.prebuilt")
            create_react_agent = getattr(prebuilt_mod, "create_react_agent")
            llm = build_chat_model(self.config.model)
            memory = MemorySaver()
            self.langchain_agent = create_react_agent(llm, tools=[], checkpointer=memory)
        except Exception:
            self.langchain_agent = None


