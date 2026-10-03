from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config



@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return recall ratio (0.0 to 1.0) depending on how many expected facts appear in answer."""
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    matches = sum(1 for exp in expected if exp.lower() in ans_lower)
    return matches / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score combining recall accuracy and structure."""
    rec = recall_points(answer, expected)
    if not answer or not answer.strip():
        return 0.0
    # Add bonus for clean structure or non-empty response when recall matches
    base_score = 0.5 * rec
    if len(answer) > 10:
        base_score += 0.3
    if any(bullet in answer for bullet in ["-", "*", "1.", "2."]):
        base_score += 0.2
    return round(min(1.0, base_score), 2)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over conversations."""
    total_agent_tokens = 0
    total_prompt_tokens = 0
    total_compactions = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    unique_users: set[str] = set()

    for conv in conversations:
        user_id = conv["user_id"]
        unique_users.add(user_id)
        thread_id = conv["id"]

        # Run conversation turns
        for turn in conv["turns"]:
            res = agent.reply(user_id, thread_id, turn)
            total_agent_tokens += res.get("agent_tokens", 0)
            total_prompt_tokens += res.get("prompt_tokens", 0)

        # Ask recall questions in a NEW thread ID to test cross-session memory
        recall_questions = conv.get("recall_questions", [])
        for idx, q_item in enumerate(recall_questions):
            recall_thread_id = f"{thread_id}_recall_{idx}"
            question = q_item["question"]
            expected = q_item["expected_contains"]

            q_res = agent.reply(user_id, recall_thread_id, question)
            total_agent_tokens += q_res.get("agent_tokens", 0)
            total_prompt_tokens += q_res.get("prompt_tokens", 0)

            ans = q_res.get("response", "")
            sc = recall_points(ans, expected)
            qual = heuristic_quality(ans, expected)
            recall_scores.append(sc)
            quality_scores.append(qual)

        total_compactions += agent.compaction_count(thread_id)

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    memory_growth = 0
    if hasattr(agent, "memory_file_size"):
        memory_growth = sum(agent.memory_file_size(u) for u in unique_users)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 2),
        response_quality=round(avg_quality, 2),
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as markdown table."""
    headers = [
        "Agent Name",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_data = [
        [
            r.agent_name,
            str(r.agent_tokens_only),
            str(r.prompt_tokens_processed),
            f"{r.recall_score * 100:.0f}%" if isinstance(r.recall_score, float) else str(r.recall_score),
            str(r.response_quality),
            str(r.memory_growth_bytes),
            str(r.compactions),
        ]
        for r in rows
    ]

    try:
        import importlib
        tab_mod = importlib.import_module("tabulate")
        tab_fn = getattr(tab_mod, "tabulate")
        return tab_fn(table_data, headers=headers, tablefmt="github")
    except Exception:
        # Fallback markdown table
        header_row = "| " + " | ".join(headers) + " |"
        sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        data_rows = ["| " + " | ".join(row) + " |" for row in table_data]
        return "\n".join([header_row, sep_row] + data_rows)



def main() -> None:
    """Run both benchmark suites and output comparison tables."""
    config = load_config(Path(__file__).resolve().parent.parent)

    std_data = load_conversations(config.data_dir / "conversations.json")
    stress_data = load_conversations(config.data_dir / "advanced_long_context.json")

    print("=" * 80)
    print(" STANDARD BENCHMARK (data/conversations.json)")
    print("=" * 80)
    baseline_std = BaselineAgent(config=config, force_offline=True)
    advanced_std = AdvancedAgent(config=config, force_offline=True)

    row_base_std = run_agent_benchmark("Baseline Agent", baseline_std, std_data, config)
    row_adv_std = run_agent_benchmark("Advanced Agent", advanced_std, std_data, config)
    print(format_rows([row_base_std, row_adv_std]))
    print()

    print("=" * 80)
    print(" LONG-CONTEXT STRESS BENCHMARK (data/advanced_long_context.json)")
    print("=" * 80)
    baseline_stress = BaselineAgent(config=config, force_offline=True)
    advanced_stress = AdvancedAgent(config=config, force_offline=True)

    row_base_stress = run_agent_benchmark("Baseline Agent", baseline_stress, stress_data, config)
    row_adv_stress = run_agent_benchmark("Advanced Agent", advanced_stress, stress_data, config)
    print(format_rows([row_base_stress, row_adv_stress]))
    print("=" * 80)



if __name__ == "__main__":
    main()

