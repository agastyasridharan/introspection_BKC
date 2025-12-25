"""
Plot success rates for introspection experiments.

Success definitions:
    - anthropic_reproduce:
        coherence AND affirmative_response_followed_by_correct_identification

    - anthropic_reproduce_binary:
        coherence AND binary_detection

    - mcq_knowledge:
        coherence AND mcq_correct

    - mcq_distinguish:
        coherence AND mcq_correct

    - open_ended_belief:
        coherence AND thinking_about_word

    - generative_distinguish:
        coherence ONLY (no correctness judge exists yet)

    - injection_strength:
        coherence AND injection_strength_correct
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


EXPERIMENT_TYPES = [
    "anthropic_reproduce",
    "anthropic_reproduce_binary",
    "anthropic_reproduce_no_base_rate",
    "mcq_knowledge",
    "mcq_distinguish",
    "open_ended_belief",
    "generative_distinguish",
    "injection_strength",
]


def coerce_bool_series(s: pd.Series) -> pd.Series:
    """Robust bool coercion for CSV columns containing bools / strings / ints."""
    # Common cases: True/False, "True"/"False", 1/0
    return s.map(lambda x: True if x is True or x == "True" or x == 1 or x == "1" else False)


def compute_success_rate(df: pd.DataFrame, experiment_type: str):
    """Compute success rate per (layer, coeff, vec_type)."""
    success_rates = defaultdict(lambda: defaultdict(dict))

    for layer in sorted(df["layer"].unique()):
        for coeff in sorted(df["coeff"].unique()):
            for vec_type in sorted(df["vec_type"].unique()):
                subset = df[
                    (df["layer"] == layer)
                    & (df["coeff"] == coeff)
                    & (df["vec_type"] == vec_type)
                ]
                if subset.empty:
                    continue

                if experiment_type in ["anthropic_reproduce", "anthropic_reproduce_no_base_rate"]:
                    successes = subset["coherence_judge"] & subset["affirmative_response_followed_by_correct_identification_judge"]

                elif experiment_type == "anthropic_reproduce_binary":
                    successes = subset["coherence_judge"] & subset["binary_detection_judge"]

                elif experiment_type in ["mcq_knowledge", "mcq_distinguish"]:
                    successes = subset["coherence_judge"] & subset["mcq_correct_judge"]

                elif experiment_type == "open_ended_belief":
                    successes = subset["coherence_judge"] & subset["thinking_about_word_judge"]

                elif experiment_type == "generative_distinguish":
                    successes = subset["coherence_judge"]

                elif experiment_type == "injection_strength":
                    successes = subset["coherence_judge"] & subset["injection_strength_correct_judge"]

                else:
                    raise ValueError(f"Unknown experiment type: {experiment_type}")

                rate = successes.fillna(False).mean()
                success_rates[layer][coeff][vec_type] = rate

    return success_rates


def plot_success_rates(success_rates, experiment_label: str, output_dir: Path):
    layers = sorted(success_rates.keys())

    coeff_vec_pairs = sorted({
        (coeff, vec_type)
        for layer_data in success_rates.values()
        for coeff, vec_dict in layer_data.items()
        for vec_type in vec_dict.keys()
    })

    plt.figure(figsize=(12, 7))
    markers = ["o", "s", "^", "D", "v", "p", "*", "h"]
    linestyles = ["-", "--", "-.", ":"]
    colors = plt.cm.tab10(range(max(1, len(coeff_vec_pairs))))

    for i, (coeff, vec_type) in enumerate(coeff_vec_pairs):
        rates = [success_rates[layer].get(coeff, {}).get(vec_type, 0.0) for layer in layers]
        plt.plot(
            layers,
            rates,
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            color=colors[i % 10],
            label=f"Coeff {coeff}, {vec_type}",
            linewidth=2,
            markersize=6,
        )

    plt.xlabel("Layer")
    plt.ylabel("Success Rate")
    plt.title(experiment_label)
    plt.legend(fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 0.8)
    plt.tight_layout()

    out = output_dir / f"success_rate_{experiment_label}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {out}")
    plt.close()


def parse_filename(csv_path: Path):
    """
    Accepts:
      - output_<exp>.csv
      - output_<exp>_assistant.csv
      - output_<exp>_all_tokens.csv

    Returns (exp_type, scope_label)
    scope_label is one of: "unscoped", "assistant", "all_tokens"
    """
    name = csv_path.stem  # no .csv
    if not name.startswith("output_"):
        return None, None

    rest = name[len("output_"):]  # remove prefix

    # Try scoped forms first
    if rest.endswith("_assistant"):
        exp = rest[: -len("_assistant")]
        return exp, "assistant"
    if rest.endswith("_all_tokens"):
        exp = rest[: -len("_all_tokens")]
        return exp, "all_tokens"

    # Unscoped
    return rest, "unscoped"


def main():
    results_dir = Path(__file__).parent.parent / "new_results"
    output_dir = Path(__file__).parent

    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir does not exist: {results_dir}")

    csv_paths = sorted(results_dir.glob("output_*.csv"))
    if not csv_paths:
        print(f"No CSVs found in {results_dir}.")
        return

    # Group by exp_type, then scope
    grouped = defaultdict(lambda: defaultdict(list))
    for p in csv_paths:
        exp, scope = parse_filename(p)
        if exp is None:
            continue
        if exp not in EXPERIMENT_TYPES:
            # Ignore unknown experiments rather than crashing
            continue
        grouped[exp][scope].append(p)

    # Plot each experiment & scope we actually have
    for exp_type in EXPERIMENT_TYPES:
        if exp_type not in grouped:
            print(f"Warning: no CSVs found for {exp_type}, skipping...")
            continue

        for scope, paths in grouped[exp_type].items():
            # If multiple files somehow exist for same (exp, scope), take the most recent by mtime
            csv_path = max(paths, key=lambda x: x.stat().st_mtime)

            print(f"\nProcessing {exp_type} ({scope}) from {csv_path.name}...")
            df = pd.read_csv(csv_path)

            # Basic sanity
            required_cols = {"layer", "coeff", "vec_type"}
            missing = required_cols - set(df.columns)
            if missing:
                print(f"Warning: {csv_path.name} missing required columns {missing}, skipping...")
                continue

            # Coerce boolean judge columns if present
            judge_cols = [
                "coherence_judge",
                "thinking_about_word_judge",
                "affirmative_response_judge",
                "affirmative_response_followed_by_correct_identification_judge",
                "binary_detection_judge",
                "mcq_correct_judge",
                "injection_strength_correct_judge",
            ]
            for col in judge_cols:
                if col in df.columns:
                    df[col] = coerce_bool_series(df[col])
                else:
                    # If missing, create False column so success logic doesn't crash
                    df[col] = False

            label = f"{exp_type}_{scope}" if scope != "unscoped" else exp_type
            success_rates = compute_success_rate(df, exp_type)
            plot_success_rates(success_rates, label, output_dir)


if __name__ == "__main__":
    main()
