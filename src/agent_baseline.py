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
    """Agent A: Baseline Agent.

    Requirements:
    - Within-session memory only (retains messages only within the current thread_id)
    - No persistent `User.md`
    - Forgets long-term facts across new threads / sessions
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting."""
        if not self.force_offline:
            chat_model = self._maybe_build_langchain_agent()
            if chat_model is not None:
                try:
                    # In live mode, baseline only passes current thread messages without profile
                    session = self.sessions.setdefault(thread_id, SessionState())
                    session.messages.append({"role": "user", "content": message})

                    turn_prompt_tokens = sum(
                        estimate_tokens(m["content"]) for m in session.messages
                    )
                    session.prompt_tokens_processed += turn_prompt_tokens

                    # Format history for chat model
                    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

                    lc_messages = [
                        SystemMessage(
                            content="You are a helpful assistant with within-session memory only. Answer concisely."
                        )
                    ]
                    for m in session.messages:
                        if m["role"] == "user":
                            lc_messages.append(HumanMessage(content=m["content"]))
                        else:
                            lc_messages.append(AIMessage(content=m["content"]))

                    response = chat_model.invoke(lc_messages)
                    reply_text = str(response.content)

                    reply_tokens = estimate_tokens(reply_text)
                    session.token_usage += reply_tokens
                    session.messages.append({"role": "assistant", "content": reply_text})

                    return {
                        "reply": reply_text,
                        "tokens": reply_tokens,
                        "prompt_tokens": turn_prompt_tokens,
                    }
                except Exception:
                    # Fallback to offline on network/api error
                    pass

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative agent response token count for one thread."""
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        """Return cumulative prompt tokens processed in one thread."""
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        """Baseline agent has no compact memory layer."""
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline behavior for baseline agent."""
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})

        # Calculate prompt context: all messages in this session so far
        turn_prompt_tokens = sum(
            estimate_tokens(m["content"]) for m in session.messages
        )
        session.prompt_tokens_processed += turn_prompt_tokens

        # Check if message is a recall question in a fresh session
        is_query = (
            message.strip().endswith("?")
            or any(
                q in message.lower()
                for q in ["nhắc lại", "tên mình là gì", "mình tên gì", "ở đâu", "nghề gì", "là ai"]
            )
        )

        if is_query and len(session.messages) <= 2:
            # Fresh thread with no prior context
            reply_text = (
                "Chào bạn! Vì đây là phiên trò chuyện mới và mình không có bộ nhớ dài hạn, "
                "mình chưa có thông tin trước đó về bạn."
            )
        else:
            # Normal conversation turn acknowledgement
            reply_text = f"Đã ghi nhận: {message[:40]}..."

        reply_tokens = estimate_tokens(reply_text)
        session.token_usage += reply_tokens
        session.messages.append({"role": "assistant", "content": reply_text})

        return {
            "reply": reply_text,
            "tokens": reply_tokens,
            "prompt_tokens": turn_prompt_tokens,
        }

    def _maybe_build_langchain_agent(self) -> Any:
        """Instantiate chat model for live execution if configured."""
        try:
            if not self.config.model.api_key and self.config.model.provider not in (
                "ollama",
                "custom",
            ):
                return None
            return build_chat_model(self.config.model)
        except Exception:
            return None
