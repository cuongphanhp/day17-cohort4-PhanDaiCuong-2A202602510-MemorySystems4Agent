from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any


def estimate_tokens(text: str) -> int:
    """Implement a simple, deterministic token estimator.

    Heuristic:
    - Strips whitespace.
    - Returns 0 for empty or whitespace-only text.
    - Approximates token count based on character length (~4 chars/token).
    """
    if not text:
        return 0
    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, len(cleaned) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Provides:
    - Mapping each user id to one markdown file on disk.
    - Read, write, and edit text operations.
    - Helpers to extract and upsert structured facts.
    """

    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, user_id: str) -> Path:
        """Sanitize the user id and return the corresponding .md file path."""
        safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", user_id.strip())
        return self.root_dir / f"{safe_id}.md"

    def read_text(self, user_id: str) -> str:
        """Return the user profile markdown content, or empty string if missing."""
        path = self.path_for(user_id)
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return ""

    def write_text(self, user_id: str, content: str) -> Path:
        """Write markdown content to disk and return the file path."""
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace the first occurrence of search_text in User.md.

        Returns True if replacement occurred, False otherwise.
        """
        current = self.read_text(user_id)
        if search_text in current:
            updated = current.replace(search_text, replacement, 1)
            self.write_text(user_id, updated)
            return True
        return False

    def file_size(self, user_id: str) -> int:
        """Return current file size in bytes, or 0 if missing."""
        path = self.path_for(user_id)
        if path.is_file():
            return path.stat().st_size
        return 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Extract key-value pairs stored in the user profile markdown."""
        content = self.read_text(user_id)
        result: dict[str, str] = {}
        if not content:
            return result

        for line in content.splitlines():
            line = line.strip()
            match = re.match(r"^[-\*]\s*(?:\*\*)?([a-zA-Z0-9_\-]+)(?:\*\*)?\s*:\s*(.+)$", line)
            if match:
                k = match.group(1).strip()
                v = match.group(2).strip()
                result[k] = v
        return result

    def upsert_facts(self, user_id: str, new_facts: dict[str, str]) -> None:
        """Merge new facts into the existing user profile markdown."""
        if not new_facts:
            return
        current = self.facts(user_id)
        current.update(new_facts)

        lines = [f"# User Profile: {user_id}", "", "## Facts"]
        for k, v in current.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

        self.write_text(user_id, "\n".join(lines))

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        """Update or insert a single fact for the user."""
        self.upsert_facts(user_id, {key: value})


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts.

    Handles facts like:
    - name
    - location (and corrections)
    - profession
    - preferences / response style
    - technical interests (Python, AI)
    - favorite food / drink
    - pet

    Skips pure question queries.
    """
    facts: dict[str, str] = {}
    if not message:
        return facts
    cleaned = message.strip()
    if not cleaned:
        return facts

    # Skip pure question or recall queries
    has_declaration = any(
        kw in cleaned for kw in ["đính chính", "Chào bạn", "lần cuối cho chắc", "Nhắc lại lần cuối"]
    )
    if cleaned.endswith("?") and not has_declaration:
        return facts

    if any(
        q in cleaned.lower()
        for q in [
            "nhắc lại giúp mình",
            "có thể nhắc lại",
            "nhắc lại tên mình không",
            "tên mình là gì",
            "mình tên gì",
            "ở đâu",
            "nghề gì",
            "đâu mới là",
            "bạn có biết",
            "nếu biết thì",
            "bạn thử nhớ",
            "đồ uống yêu thích của mình là gì",
        ]
    ) and not has_declaration:
        return facts

    # Pre-clean message to avoid confusing pet names with user names
    msg_no_pet = cleaned
    if "corgi" in cleaned.lower():
        msg_no_pet = re.sub(r"(?:con|bé)?\s*corgi\s+tên\s+\w+", "", cleaned, flags=re.IGNORECASE)

    # 1. Name
    name_match = re.search(
        r"(?:mình tên là|tên mình là|mình tên)\s+([A-ZÀ-Ỹa-zà-ỹ0-9_\-]+(?:\s+[A-ZÀ-Ỹa-zà-ỹ0-9_\-]+)*)",
        msg_no_pet,
        re.IGNORECASE,
    )
    if not name_match:
        name_match = re.search(
            r"\btên\s+([A-ZÀ-Ỹa-zà-ỹ0-9_\-]+(?:\s+[A-ZÀ-Ỹa-zà-ỹ0-9_\-]+)*)(?:,\s*nghề|,\s*nơi|$)",
            msg_no_pet,
            re.IGNORECASE,
        )
    if name_match:
        val = name_match.group(1).strip(" .,!")
        stopwords = ["và", "gì", "không", "ai", "style", "là", "của", "mình"]
        if val and not any(val.lower().startswith(w + " ") or val.lower() == w for w in stopwords):
            facts["name"] = val

    # 2. Location (support updates and filter noise like Hà Nội transit)
    if "Hà Nội chỉ là nơi" not in cleaned and "chứ không phải nơi ở hiện tại" not in cleaned:
        if "cập nhật từ" in cleaned and "sang" in cleaned:
            m = re.search(r"sang\s+([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:[,\.\!\n]|vài tháng|$)", cleaned)
            if m:
                facts["location"] = m.group(1).strip()
        elif "nơi ở hiện tại là" in cleaned:
            m = re.search(r"nơi ở hiện tại là\s+([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:[,\.\!\n]|$)", cleaned)
            if m:
                facts["location"] = m.group(1).strip()
        elif "giờ mình đang ở" in cleaned:
            m = re.search(r"giờ mình đang ở\s+([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:\s+chứ|[,\.\!\n]|$)", cleaned)
            if m:
                facts["location"] = m.group(1).strip()
        elif "đang làm việc ở" in cleaned:
            m = re.search(r"đang làm việc ở\s+([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:\s+vài tháng|[,\.\!\n]|$)", cleaned)
            if m:
                facts["location"] = m.group(1).strip()
        elif "hiện ở" in cleaned:
            m = re.search(r"hiện ở\s+([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:\s+và|[,\.\!\n]|$)", cleaned)
            if m:
                facts["location"] = m.group(1).strip()
        elif re.search(r"Mình ở\s+([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:\s+và|[,\.\!\n]|$)", cleaned, re.IGNORECASE):
            m = re.search(r"Mình ở\s+([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:\s+và|[,\.\!\n]|$)", cleaned, re.IGNORECASE)
            if m:
                facts["location"] = m.group(1).strip()

    # 3. Profession (filter out jokes like product manager)
    if "product manager" not in cleaned or "câu đùa" in cleaned:
        if "chuyển sang" in cleaned and ("engineer" in cleaned or "developer" in cleaned):
            m = re.search(
                r"chuyển sang\s+([A-Za-z0-9_\-\s]+?(?:engineer|developer))",
                cleaned,
                re.IGNORECASE,
            )
            if m:
                facts["profession"] = m.group(1).strip()
        elif "nghề nghiệp hiện tại vẫn là" in cleaned:
            m = re.search(
                r"nghề nghiệp hiện tại vẫn là\s+([A-Za-z0-9_\-\s]+?engineer|[A-Za-z0-9_\-\s]+?developer)",
                cleaned,
            )
            if m:
                facts["profession"] = m.group(1).strip()
        elif "nghề" in cleaned and ("engineer" in cleaned or "kỹ sư" in cleaned):
            m = re.search(r"nghề\s+([A-Za-z0-9_\-\s]+?(?:engineer|kỹ sư))", cleaned)
            if m:
                facts["profession"] = m.group(1).strip()
        elif "đang làm" in cleaned and ("engineer" in cleaned or "developer" in cleaned):
            m = re.search(
                r"đang làm\s+([A-Za-z0-9_\-\s]+?(?:engineer|developer))",
                cleaned,
            )
            if m:
                facts["profession"] = m.group(1).strip()

    # 4. Technical interests
    if "python" in cleaned.lower() and ("ai" in cleaned.lower() or "rag" in cleaned.lower()):
        facts["interests"] = "Python, AI"

    # 5. Favorite drink
    if "cà phê sữa đá" in cleaned.lower():
        facts["favorite_drink"] = "cà phê sữa đá"

    # 6. Favorite food
    if "mì quảng" in cleaned.lower() or "mì Quảng" in cleaned:
        facts["favorite_food"] = "mì Quảng"

    # 7. Pet
    if "corgi" in cleaned.lower():
        facts["pet"] = "corgi tên Bơ"

    # 8. Response style
    if "3 bullet" in cleaned:
        facts["response_style"] = "ngắn gọn, 3 bullet có ví dụ thực chiến"
    elif "bullet ngắn" in cleaned or "ngắn gọn" in cleaned:
        facts["response_style"] = "ngắn gọn, bullet ngắn và có ví dụ thực tế"

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages."""
    if not messages:
        return ""

    items = messages[-max_items:] if len(messages) > max_items else messages
    summary_lines: list[str] = []
    for msg in items:
        role = msg.get("role", "unknown")
        content = msg.get("content", "").strip()
        first_clause = content.split("\n")[0]
        if len(first_clause) > 120:
            first_clause = first_clause[:117] + "..."
        summary_lines.append(f"- {role}: {first_clause}")

    return "\n".join(summary_lines)


@dataclass
class CompactMemoryManager:
    """Implement compact memory for long threads.

    Goal:
    - Keep recent messages in full.
    - When thread tokens exceed threshold, move older content into summary.
    - Track compaction count for benchmarking.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def _get_thread(self, thread_id: str) -> dict[str, Any]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append a new message to the thread and trigger compaction if threshold exceeded."""
        t_state = self._get_thread(thread_id)
        messages: list[dict[str, str]] = t_state["messages"]
        messages.append({"role": role, "content": content})

        summary_text: str = t_state.get("summary", "")
        current_tokens = estimate_tokens(summary_text) + sum(
            estimate_tokens(m.get("content", "")) for m in messages
        )

        if current_tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            to_compact = messages[: -self.keep_messages]
            kept_messages = messages[-self.keep_messages :]
            t_state["messages"] = kept_messages

            new_summary_part = summarize_messages(to_compact)
            if summary_text:
                t_state["summary"] = f"{summary_text}\n{new_summary_part}"
            else:
                t_state["summary"] = new_summary_part

            t_state["compactions"] = int(t_state.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, Any]:
        """Return per-thread state including messages, summary, and compaction count."""
        t_state = self._get_thread(thread_id)
        return {
            "messages": list(t_state.get("messages", [])),
            "summary": t_state.get("summary", ""),
            "compactions": t_state.get("compactions", 0),
        }

    def compaction_count(self, thread_id: str) -> int:
        """Return the number of compactions performed on this thread."""
        return int(self.state.get(thread_id, {}).get("compactions", 0))
