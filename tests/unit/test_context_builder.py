from brain.context_builder import ContextBudget, build_context
from brain.adapter import MemoryRecord


def test_context_builder_delimits_memory_and_accounts_for_tokens() -> None:
    result = build_context(
        system_instructions="You are a helpful assistant.",
        current_user_message="What database do we use?",
        recent_conversation=[
            {"role": "user", "content": "Our production database is PostgreSQL."},
            {"role": "assistant", "content": "I will remember that."},
        ],
        memories=[MemoryRecord(memory_id="db", text="Production uses PostgreSQL.")],
        budget=ContextBudget(max_tokens=100, recent_turn_budget=100, memory_budget=100),
    )

    assert result.messages[-1] == {"role": "user", "content": "What database do we use?"}
    assert "<retrieved_memory>" in result.messages[1]["content"]
    assert result.stats.selected_memory_count == 1
    assert result.stats.raw_history_tokens > 0


def test_recent_history_is_selected_from_the_end() -> None:
    result = build_context(
        system_instructions="",
        current_user_message="current",
        recent_conversation=[
            {"role": "user", "content": "old " * 20},
            {"role": "assistant", "content": "new"},
        ],
        memories=[],
        budget=ContextBudget(recent_turn_budget=2),
    )

    contents = [message["content"] for message in result.messages]
    assert "new" in contents
    assert not any(content.startswith("old") for content in contents)
