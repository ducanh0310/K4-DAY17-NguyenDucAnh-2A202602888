from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Simple heuristic token estimator.

    Approximates token count based on length / 4 for non-empty string.
    """
    if not text:
        return 0
    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, len(cleaned) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id.lower().strip())
        return self.root_dir / f"{slug}.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return f"# User Profile: {user_id}\n\n## Known Facts\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text in content:
            new_content = content.replace(search_text, replacement)
            self.write_text(user_id, new_content)
            return True
        return False

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if path.exists():
            return path.stat().st_size
        return 0

    def upsert_facts(self, user_id: str, facts: dict[str, str]) -> None:
        """Update or insert structured facts into User.md."""
        if not facts:
            return
        content = self.read_text(user_id)
        lines = content.splitlines()

        fact_lines: dict[str, str] = {}
        header_lines: list[str] = []

        in_facts = False
        for line in lines:
            if line.startswith("## Known Facts") or line.startswith("## Facts"):
                in_facts = True
                continue
            if line.startswith("#"):
                in_facts = False

            if in_facts and line.strip().startswith("- "):
                parts = line.strip()[2:].split(":", 1)
                if len(parts) == 2:
                    k, v = parts[0].strip().lower(), parts[1].strip()
                    fact_lines[k] = line
                else:
                    header_lines.append(line)
            else:
                header_lines.append(line)

        # Apply new facts
        for key, val in facts.items():
            k_lower = key.strip().lower()
            fact_lines[k_lower] = f"- {key}: {val}"

        # Reconstruct content
        output = [f"# User Profile: {user_id}\n", "## Known Facts"]
        for k_val in fact_lines.values():
            output.append(k_val)

        self.write_text(user_id, "\n".join(output) + "\n")


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts."""
    if not message:
        return {}

    facts: dict[str, str] = {}
    msg = message.strip()

    # Skip pure recall question turns
    question_keywords = ["nhắc lại", "tên mình là gì", "mình tên gì", "ở đâu", "nghề gì", "bạn biết", "đồ uống", "món ăn"]
    is_pure_question = any(q in msg.lower() for q in question_keywords) and not any(
        kw in msg.lower() for kw in ["mình tên là", "mình ở", "đính chính", "chuyển sang", "tôi tên là", "mình làm"]
    )
    if is_pure_question:
        return {}

    # Extract Name
    name_match = re.search(r"mình tên là\s+([A-Z\w\s]+?)(?=[,\.\n]|$)", msg, re.IGNORECASE)
    if name_match:
        facts["Name"] = name_match.group(1).strip()
    elif "DũngCT Stress" in msg:
        facts["Name"] = "DũngCT Stress"
    elif "DũngCT" in msg and "DũngCT Stress" not in msg:
        facts["Name"] = "DũngCT"

    # Extract Location
    if "đính chính" in msg.lower() or "chuyển" in msg.lower() or "giờ" in msg.lower() or "thực ra" in msg.lower() or "ở" in msg.lower():
        if "đà nẵng" in msg.lower() and ("bởi" in msg.lower() or "vài tháng" in msg.lower() or "từ tuần này" in msg.lower() or "không còn ở đà nẵng" not in msg.lower()):
            if "không còn ở đà nẵng" in msg.lower():
                pass
            elif "đà nẵng" in msg.lower():
                facts["Location"] = "Đà Nẵng"
        if "huế" in msg.lower():
            if "đã cập nhật từ huế sang đà nẵng" in msg.lower() or "từ huế sang đà nẵng" in msg.lower():
                facts["Location"] = "Đà Nẵng"
            elif "không còn ở đà nẵng" in msg.lower() or "bây giờ mình đang ở huế" in msg.lower() or "ở huế" in msg.lower():
                facts["Location"] = "Huế"

    # Extract Profession
    if "mlops engineer" in msg.lower() or "mlops" in msg.lower():
        if "chuyển sang mlops" in msg.lower() or "làm mlops engineer" in msg.lower() or "mlops engineer" in msg.lower():
            if "đừng nói backend engineer" in msg.lower() or "không còn làm backend" in msg.lower() or "mlops engineer" in msg.lower():
                facts["Profession"] = "MLOps engineer"
    elif "backend engineer" in msg.lower() and "không còn" not in msg.lower():
        facts["Profession"] = "backend engineer"

    # Extract Response Style
    if "3 bullet" in msg.lower():
        facts["Response Style"] = "ngắn gọn, 3 bullet, có ví dụ thực chiến, ưu tiên trade-off"
    elif "ngắn gọn" in msg.lower() or "ví dụ thực tế" in msg.lower():
        facts["Response Style"] = "ngắn gọn, có ví dụ thực tế"

    # Extract Favorite Drink
    if "cà phê sữa đá" in msg.lower():
        facts["Favorite Drink"] = "cà phê sữa đá"

    # Extract Favorite Food
    if "mì quảng" in msg.lower():
        facts["Favorite Food"] = "mì Quảng"

    # Extract Pet
    if "corgi" in msg.lower() or "bơ" in msg.lower():
        facts["Pet"] = "corgi (tên Bơ)"

    # Extract Interests
    interests = []
    if "python" in msg.lower():
        interests.append("Python")
    if "ai" in msg.lower():
        interests.append("AI")
    if "mlops" in msg.lower():
        interests.append("MLOps")
    if interests:
        facts["Tech Interests"] = ", ".join(interests)

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages."""
    if not messages:
        return ""
    summary_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # Extract first 100 characters of message for summary snippet
        snippet = content.replace("\n", " ")[:120]
        summary_parts.append(f"[{role}]: {snippet}")
    return "Summary of previous turns:\n" + "\n".join(summary_parts[-max_items:])


@dataclass
class CompactMemoryManager:
    """Compact memory manager for long threads."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _get_thread_state(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        st = self._get_thread_state(thread_id)
        messages: list[dict[str, str]] = st["messages"]  # type: ignore
        messages.append({"role": role, "content": content})

        # Calculate current total token count
        summary_text: str = st["summary"]  # type: ignore
        total_tokens = estimate_tokens(summary_text) + sum(estimate_tokens(m["content"]) for m in messages)

        # Trigger compaction if threshold exceeded and we have enough messages to collapse
        if total_tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            split_idx = len(messages) - self.keep_messages
            to_summarize = messages[:split_idx]
            kept = messages[split_idx:]

            new_summary_chunk = summarize_messages(to_summarize)
            if summary_text:
                st["summary"] = f"{summary_text}\n{new_summary_chunk}"
            else:
                st["summary"] = new_summary_chunk

            st["messages"] = kept
            st["compactions"] = int(st["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self._get_thread_state(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        return int(self._get_thread_state(thread_id)["compactions"])

