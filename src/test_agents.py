from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for testing."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    dummy_provider = ProviderConfig(provider="openai", model_name="gpt-4o-mini")
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=50,
        compact_keep_messages=2,
        model=dummy_provider,
        judge_model=dummy_provider,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, read, updated, and edited."""
    store = UserProfileStore(root_dir=tmp_path / "profiles")
    user_id = "test_user_01"

    # Test initial write
    initial_content = "# User Profile: test_user_01\n- Location: Đà Nẵng\n"
    written_path = store.write_text(user_id, initial_content)
    assert written_path.exists()

    # Test read
    content = store.read_text(user_id)
    assert "Đà Nẵng" in content

    # Test edit
    edited = store.edit_text(user_id, "Đà Nẵng", "Huế")
    assert edited is True
    updated_content = store.read_text(user_id)
    assert "Huế" in updated_content
    assert "Đà Nẵng" not in updated_content

    # Test file size
    assert store.file_size(user_id) > 0


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""
    compact_mgr = CompactMemoryManager(threshold_tokens=40, keep_messages=2)
    thread_id = "test_thread_compact"

    # Append short message 1 & 2
    compact_mgr.append(thread_id, "user", "Message 1 short text.")
    compact_mgr.append(thread_id, "user", "Message 2 short text.")
    assert compact_mgr.compaction_count(thread_id) == 0

    # Append long message 3 & 4 exceeding threshold_tokens (40 tokens ~ 160 chars)
    compact_mgr.append(thread_id, "user", "This is a very long message 3 that contains a lot of text to push token count over the compact threshold limit quickly.")
    compact_mgr.append(thread_id, "user", "This is another long message 4 that forces compaction to run and collapse older messages into summary.")

    assert compact_mgr.compaction_count(thread_id) > 0
    ctx = compact_mgr.context(thread_id)
    assert bool(ctx["summary"]) is True
    assert len(ctx["messages"]) <= 2  # type: ignore


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    user_id = "dungct_recall_test"
    thread_1 = "thread_session_1"
    thread_2 = "thread_session_2"

    # Session 1: User introduces info
    info_msg = "Chào bạn, mình tên là DũngCT và đồ uống yêu thích là cà phê sữa đá."
    baseline.reply(user_id, thread_1, info_msg)
    advanced.reply(user_id, thread_1, info_msg)

    # Session 2: Fresh thread_id asking recall question
    question = "Mình tên gì và đồ uống yêu thích là gì?"
    base_res = baseline.reply(user_id, thread_2, question)["response"]
    adv_res = advanced.reply(user_id, thread_2, question)["response"]

    # Baseline should NOT remember across sessions
    assert "DũngCT" not in base_res or "cà phê sữa đá" not in base_res

    # Advanced MUST remember across sessions from User.md
    assert "DũngCT" in adv_res
    assert "cà phê sữa đá" in adv_res


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    config = make_config(tmp_path)
    # Set a small threshold for advanced compaction
    config.compact_threshold_tokens = 30
    config.compact_keep_messages = 2

    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    user_id = "stress_user"
    thread_id = "long_stress_thread"

    # Send 10 long turns
    long_msg = "Đây là một tin nhắn rất dài nhằm kiểm tra khả năng nén bộ nhớ của compact memory manager khi hội thoại kéo dài liên tục qua nhiều lượt."
    
    last_base_prompt_tokens = 0
    last_adv_prompt_tokens = 0

    for i in range(10):
        b_res = baseline.reply(user_id, thread_id, f"Turn {i}: {long_msg}")
        a_res = advanced.reply(user_id, thread_id, f"Turn {i}: {long_msg}")
        last_base_prompt_tokens = b_res["prompt_tokens"]
        last_adv_prompt_tokens = a_res["prompt_tokens"]

    # Advanced should have triggered compaction
    assert advanced.compaction_count(thread_id) > 0

    # In turn 10, baseline prompt tokens should be much larger than advanced prompt tokens
    assert last_adv_prompt_tokens < last_base_prompt_tokens

