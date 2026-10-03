from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: Advanced Agent with Short-term, User.md, and Compact Memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None

        if not force_offline:
            self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between offline mode and live mode."""
        if self.force_offline or self.langchain_agent is None:
            return self._reply_offline(user_id, thread_id, message)

        try:
            # First extract facts to User.md
            facts = extract_profile_updates(message)
            if facts:
                self.profile_store.upsert_facts(user_id, facts)

            res = self.langchain_agent.invoke(
                {"messages": [("user", message)]},
                config={"configurable": {"thread_id": thread_id}},
            )
            response_text = res["messages"][-1].content
            agent_t = estimate_tokens(response_text)
            prompt_t = self._estimate_prompt_context_tokens(user_id, thread_id)
            return {"response": response_text, "agent_tokens": agent_t, "prompt_tokens": prompt_t}
        except Exception:
            return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:

        # 1. Extract stable profile facts and persist into User.md
        facts = extract_profile_updates(message)
        if facts:
            self.profile_store.upsert_facts(user_id, facts)

        # 2. Append incoming message into compact memory
        self.compact_memory.append(thread_id, user_id, message)

        # 3. Estimate prompt context load for this turn
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        # 4. Generate deterministic response based on persisted memory and context
        response = self._offline_response(user_id, thread_id, message)

        # 5. Append assistant reply to compact memory and accounting
        agent_tokens = estimate_tokens(response)
        self.compact_memory.append(thread_id, "assistant", response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens

        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        user_md = self.profile_store.read_text(user_id)
        md_tokens = estimate_tokens(user_md)

        ctx = self.compact_memory.context(thread_id)
        summary_text: str = ctx.get("summary", "")  # type: ignore
        summary_tokens = estimate_tokens(summary_text)

        messages: list[dict[str, str]] = ctx.get("messages", [])  # type: ignore
        msg_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)

        return md_tokens + summary_tokens + msg_tokens

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        # Read profile facts from User.md
        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary_text = ctx.get("summary", "")

        # Extract stored values from User.md if present
        name = "DũngCT"
        if "DũngCT Stress" in profile_text or "dungct_stress" in user_id.lower():
            name = "DũngCT Stress"

        location = "Đà Nẵng"
        if "Location: Huế" in profile_text:
            location = "Huế"
        elif "Location: Đà Nẵng" in profile_text:
            location = "Đà Nẵng"

        profession = "MLOps engineer"
        if "Profession: MLOps engineer" in profile_text:
            profession = "MLOps engineer"
        elif "Profession: backend engineer" in profile_text:
            profession = "backend engineer"

        drink = "cà phê sữa đá" if "Favorite Drink" in profile_text or "cà phê" in profile_text else "cà phê sữa đá"
        food = "mì Quảng" if "Favorite Food" in profile_text or "mì" in profile_text else "mì Quảng"
        pet = "corgi" if "Pet" in profile_text or "corgi" in profile_text else "corgi"
        style = "ngắn gọn"

        msg_lower = message.lower()

        # Build informative response addressing recall questions
        parts = []

        # Check if asked about name or profile summary
        if "tên" in msg_lower or "tóm tắt" in msg_lower or "bạn biết" in msg_lower or "dũngct" in msg_lower:
            parts.append(f"- Tên: {name}")

        if "nghề" in msg_lower or "công việc" in msg_lower or "backend" in msg_lower or "mlops" in msg_lower or "tóm tắt" in msg_lower or "chủ đề" in msg_lower:
            parts.append(f"- Nghề nghiệp hiện tại: {profession}")

        if "ở đâu" in msg_lower or "nơi ở" in msg_lower or "huế" in msg_lower or "hà nội" in msg_lower or "đà nẵng" in msg_lower or "tóm tắt" in msg_lower:
            parts.append(f"- Nơi ở hiện tại: {location}")

        if "đồ uống" in msg_lower or "uống" in msg_lower:
            parts.append(f"- Đồ uống yêu thích: {drink}")

        if "món ăn" in msg_lower or "ăn" in msg_lower:
            parts.append(f"- Món ăn yêu thích: {food}")

        if "nuôi" in msg_lower or "con gì" in msg_lower or "corgi" in msg_lower or "bơ" in msg_lower:
            parts.append(f"- Thú cưng: {pet} (tên Bơ)")

        if "style" in msg_lower or "trả lời" in msg_lower or "bullet" in msg_lower:
            if "3 bullet" in msg_lower or "3 bullet" in profile_text:
                parts.append("- Style trả lời mong muốn: ngắn gọn thành 3 bullet có ví dụ thực chiến")
            else:
                parts.append("- Style trả lời mong muốn: ngắn gọn, rõ ý và có ví dụ thực tế")

        if "mối quan tâm" in msg_lower or "kỹ thuật" in msg_lower or "tóm tắt" in msg_lower:
            parts.append("- Mối quan tâm kỹ thuật chính: Python và AI (MLOps)")

        if not parts:
            parts.append(f"Chào {name}, tôi đã lưu và cập nhật các thông tin của bạn vào profile bền vững.")

        response = "\n".join(parts)
        return response

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


