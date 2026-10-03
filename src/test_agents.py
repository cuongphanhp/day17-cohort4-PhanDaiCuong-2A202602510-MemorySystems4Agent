from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests."""
    cfg = load_config()
    cfg.state_dir = tmp_path / "state"
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    cfg.compact_threshold_tokens = 60
    cfg.compact_keep_messages = 2
    return cfg


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "test_user"

    # Write initial profile
    initial_content = "# User Profile: test_user\n\n## Facts\n- name: Alice\n- location: Hanoi\n"
    store.write_text(user_id, initial_content)

    # Read profile
    read_back = store.read_text(user_id)
    assert "Alice" in read_back
    assert "Hanoi" in read_back
    assert store.file_size(user_id) > 0

    # Edit profile
    success = store.edit_text(user_id, "Hanoi", "Da Nang")
    assert success is True
    updated = store.read_text(user_id)
    assert "Da Nang" in updated
    assert "Hanoi" not in updated

    # Edit non-existent text
    fail = store.edit_text(user_id, "NonExistentLocation", "Tokyo")
    assert fail is False


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""
    cfg = make_config(tmp_path)
    mgr = CompactMemoryManager(
        threshold_tokens=cfg.compact_threshold_tokens,
        keep_messages=cfg.compact_keep_messages,
    )

    thread_id = "test_thread"
    for i in range(10):
        mgr.append(
            thread_id,
            "user",
            f"Message turn number {i}: This is an extended message intended to consume tokens and trigger memory compaction.",
        )

    # Verify compaction occurred
    assert mgr.compaction_count(thread_id) > 0
    ctx = mgr.context(thread_id)
    assert len(ctx["messages"]) == cfg.compact_keep_messages
    assert ctx["summary"] != ""


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""
    cfg = make_config(tmp_path)
    baseline = BaselineAgent(cfg, force_offline=True)
    advanced = AdvancedAgent(cfg, force_offline=True)

    user_id = "dungct"
    # Thread 1: provide information
    turn1 = "Chào bạn, mình tên là DũngCT. Đồ uống yêu thích là cà phê sữa đá."
    baseline.reply(user_id, "thread_1", turn1)
    advanced.reply(user_id, "thread_1", turn1)

    # Thread 2: ask in a brand new session
    query = "Mình tên gì và đồ uống yêu thích là gì?"
    base_res = baseline.reply(user_id, "thread_2", query)
    adv_res = advanced.reply(user_id, "thread_2", query)

    # Baseline should NOT remember across sessions
    assert "cà phê sữa đá" not in base_res["reply"]

    # Advanced should remember from User.md
    assert "DũngCT" in adv_res["reply"]
    assert "cà phê sữa đá" in adv_res["reply"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    cfg = make_config(tmp_path)
    baseline = BaselineAgent(cfg, force_offline=True)
    advanced = AdvancedAgent(cfg, force_offline=True)

    user_id = "stress_user"
    thread_id = "long_conversation_thread"

    # Send multiple long turns
    for i in range(12):
        msg = (
            f"Turn {i}: Đoạn văn dài dùng để kiểm tra hiệu năng token của bộ nhớ agent. "
            "Chúng ta bàn về các vấn đề kỹ thuật liên quan đến MLOps, pipeline dữ liệu, "
            "và việc tối ưu prompt context size khi hội thoại kéo dài qua nhiều lượt hỏi đáp."
        )
        baseline.reply(user_id, thread_id, msg)
        advanced.reply(user_id, thread_id, msg)

    # Advanced must trigger compaction
    assert advanced.compaction_count(thread_id) > 0

    base_prompt_load = baseline.prompt_token_usage(thread_id)
    adv_prompt_load = advanced.prompt_token_usage(thread_id)

    # Advanced prompt load must be lower than baseline prompt load
    assert adv_prompt_load < base_prompt_load, (
        f"Expected advanced prompt load ({adv_prompt_load}) < baseline ({base_prompt_load})"
    )


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tpath = Path(tmp)
        print("▶ Running test_user_markdown_read_write_edit...")
        test_user_markdown_read_write_edit(tpath)
        print("▶ Running test_compact_trigger...")
        test_compact_trigger(tpath)
        print("▶ Running test_cross_session_recall...")
        test_cross_session_recall(tpath)
        print("▶ Running test_compact_reduces_prompt_load_on_long_thread...")
        test_compact_reduces_prompt_load_on_long_thread(tpath)
        print("\n✅ ALL 4 TESTS IN test_agents.py PASSED SUCCESSFULLY!")
