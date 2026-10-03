from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


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
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0 to 1 depending on how many expected facts appear."""
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    matched = sum(1 for exp in expected if exp.lower() in ans_lower)
    return round(matched / len(expected), 3)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score for offline/online evaluation."""
    rec = recall_points(answer, expected)
    if not answer.strip():
        return 0.0

    length_bonus = 0.1 if len(answer.strip()) >= 20 else 0.0
    structure_bonus = 0.1 if ("-" in answer or "\n" in answer or ":" in answer) else 0.0
    score = (rec * 0.8) + length_bonus + structure_bonus
    return min(1.0, round(score, 3))


def run_agent_benchmark(
    agent_name: str,
    agent: BaselineAgent | AdvancedAgent,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    """Evaluate one agent over conversations and recall questions."""
    total_agent_tokens = 0
    total_prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    # Get initial memory size
    user_ids = {c["user_id"] for c in conversations}
    initial_memory = 0
    if hasattr(agent, "memory_file_size"):
        initial_memory = sum(agent.memory_file_size(u) for u in user_ids)

    # 1. Feed conversation turns
    for conv in conversations:
        conv_id = conv["id"]
        user_id = conv["user_id"]
        for turn in conv["turns"]:
            res = agent.reply(user_id=user_id, thread_id=conv_id, message=turn)
            total_agent_tokens += res.get("tokens", 0)
            total_prompt_tokens += res.get("prompt_tokens", 0)

        # 2. Ask recall questions in fresh sessions
        for q_idx, q in enumerate(conv.get("recall_questions", [])):
            recall_thread = f"{conv_id}_recall_{q_idx}"
            q_res = agent.reply(
                user_id=user_id,
                thread_id=recall_thread,
                message=q["question"],
            )
            total_agent_tokens += q_res.get("tokens", 0)
            total_prompt_tokens += q_res.get("prompt_tokens", 0)

            rec = recall_points(q_res["reply"], q.get("expected_contains", []))
            qual = heuristic_quality(q_res["reply"], q.get("expected_contains", []))
            recall_scores.append(rec)
            quality_scores.append(qual)

    # 3. Calculate final memory and compactions
    final_memory = 0
    if hasattr(agent, "memory_file_size"):
        final_memory = sum(agent.memory_file_size(u) for u in user_ids)
    memory_growth = max(0, final_memory - initial_memory)

    total_compactions = 0
    if hasattr(agent, "compact_memory"):
        for state in agent.compact_memory.state.values():
            total_compactions += int(state.get("compactions", 0))

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 3),
        response_quality=round(avg_quality, 3),
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Print a clean markdown table of benchmark rows."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        lines.append(
            f"| {r.agent_name} | {r.agent_tokens_only:,} | {r.prompt_tokens_processed:,} | "
            f"{r.recall_score * 100:.1f}% | {r.response_quality * 100:.1f}% | "
            f"{r.memory_growth_bytes:,} | {r.compactions} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Run both standard benchmark and long-context stress benchmark."""
    root_dir = Path(__file__).resolve().parent.parent
    config = load_config(root_dir)

    std_data_path = config.data_dir / "conversations.json"
    stress_data_path = config.data_dir / "advanced_long_context.json"

    print("================================================================================")
    print("🚀 DAY 17: MEMORY SYSTEMS BENCHMARK (BASELINE vs ADVANCED)")
    print("================================================================================\n")

    # 1. Standard Benchmark
    print("▶ Running Suite 1: Standard Benchmark (data/conversations.json)...")
    std_convs = load_conversations(std_data_path)

    import shutil

    # Use isolated state directories for clean measurement
    base_std_cfg = load_config(root_dir)
    base_std_cfg.state_dir = root_dir / "state" / "bench_base_std"
    shutil.rmtree(base_std_cfg.state_dir, ignore_errors=True)

    adv_std_cfg = load_config(root_dir)
    adv_std_cfg.state_dir = root_dir / "state" / "bench_adv_std"
    shutil.rmtree(adv_std_cfg.state_dir, ignore_errors=True)

    base_agent = BaselineAgent(base_std_cfg, force_offline=True)
    adv_agent = AdvancedAgent(adv_std_cfg, force_offline=True)

    row_base_std = run_agent_benchmark("Baseline Agent", base_agent, std_convs, base_std_cfg)
    row_adv_std = run_agent_benchmark("Advanced Agent", adv_agent, std_convs, adv_std_cfg)

    print("\n### Standard Benchmark Results:")
    print(format_rows([row_base_std, row_adv_std]))

    # 2. Long-Context Stress Benchmark
    print("\n\n▶ Running Suite 2: Long-Context Stress Benchmark (data/advanced_long_context.json)...")
    stress_convs = load_conversations(stress_data_path)

    base_stress_cfg = load_config(root_dir)
    base_stress_cfg.state_dir = root_dir / "state" / "bench_base_stress"
    shutil.rmtree(base_stress_cfg.state_dir, ignore_errors=True)

    adv_stress_cfg = load_config(root_dir)
    adv_stress_cfg.state_dir = root_dir / "state" / "bench_adv_stress"
    shutil.rmtree(adv_stress_cfg.state_dir, ignore_errors=True)
    adv_stress_cfg.compact_threshold_tokens = 600
    adv_stress_cfg.compact_keep_messages = 4

    base_stress_agent = BaselineAgent(base_stress_cfg, force_offline=True)
    adv_stress_agent = AdvancedAgent(adv_stress_cfg, force_offline=True)

    row_base_stress = run_agent_benchmark(
        "Baseline Agent", base_stress_agent, stress_convs, base_stress_cfg
    )
    row_adv_stress = run_agent_benchmark(
        "Advanced Agent", adv_stress_agent, stress_convs, adv_stress_cfg
    )

    print("\n### Long-Context Stress Benchmark Results:")
    print(format_rows([row_base_stress, row_adv_stress]))

    print("\n================================================================================")
    print("✅ BENCHMARK COMPLETED SUCCESSFULLY")
    print("================================================================================")


if __name__ == "__main__":
    main()
