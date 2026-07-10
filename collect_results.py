"""
collect_results.py
Reads all per-run CSVs from results/raw/ and produces:
  1. results/summary.csv — master table with config columns
  2. results/plots/ — learning curve comparisons
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "results", "raw")
PLOTS_DIR = os.path.join(BASE_DIR, "results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def entropy(counts):
    """Compute entropy of action distribution."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counts])
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def process_run(filepath):
    """Read a single CSV and return summary metrics."""
    scores, losses, lengths, actions, dists = [], [], [], [], []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row["score"]))
            if row["loss"]:
                losses.append(float(row["loss"]))
            lengths.append(int(row["ep_length"]))
            actions.append([int(row[f"action_{i}"]) for i in range(5)])
            dists.append(float(row["shark_dist_avg"]))

    n = len(scores)
    last100 = slice(max(0, n - 100), n) if n >= 100 else slice(0, n)

    act_total = [sum(a[i] for a in actions[last100]) for i in range(5)]
    return {
        "final_score": np.mean(scores[last100]),
        "final_loss": np.mean(losses[last100]) if losses else 0,
        "mean_survival": np.mean(lengths[last100]),
        "action_entropy": entropy(act_total),
        "mean_shark_dist": np.mean(dists[last100]) if dists else 0,
        "scores": scores,
        "losses": losses,
    }


def plot_learning_curves(all_data):
    """Plot score learning curves grouped by config."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, runs in all_data.items():
        scores_arrays = [np.array(r["scores"]) for r in runs]
        min_len = min(len(s) for s in scores_arrays)
        truncated = [s[:min_len] for s in scores_arrays]
        mean_curve = np.mean(truncated, axis=0)
        std_curve = np.std(truncated, axis=0)
        x = np.arange(1, min_len + 1)

        axes[0].plot(x, mean_curve, label=f"{label} (n={len(runs)})")
        axes[0].fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Avg Score")
    axes[0].set_title("Learning Curves (mean ± std)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "learning_curves.png"), dpi=150)
    plt.close()


def main():
    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    if not raw_files:
        print("No CSV files found in results/raw/. Run run_ablation.py first.")
        return

    summary_rows = []
    all_data = {}

    for fname in raw_files:
        parts = fname.replace(".csv", "").split("_")
        school_size = int(parts[0].replace("school", ""))
        ray_count = int(parts[1].replace("rays", ""))
        seed = int(parts[2].replace("seed", ""))
        config_label = f"S{school_size}R{ray_count}"

        filepath = os.path.join(RAW_DIR, fname)
        metrics = process_run(filepath)

        summary_rows.append({
            "config_name": config_label,
            "school_size": school_size,
            "ray_count": ray_count,
            "seed": seed,
            "final_score": round(metrics["final_score"], 2),
            "final_loss": round(metrics["final_loss"], 4),
            "mean_survival": round(metrics["mean_survival"], 1),
            "action_entropy": round(metrics["action_entropy"], 3),
            "mean_shark_dist": round(metrics["mean_shark_dist"], 2),
        })

        key = config_label
        if key not in all_data:
            all_data[key] = []
        all_data[key].append(metrics)

    # Write summary CSV
    summary_path = os.path.join(BASE_DIR, "results", "summary.csv")
    with open(summary_path, "w", newline="") as f:
        fieldnames = [
            "config_name", "school_size", "ray_count", "seed",
            "final_score", "final_loss", "mean_survival",
            "action_entropy", "mean_shark_dist",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(summary_rows, key=lambda r: (r["school_size"], r["ray_count"], r["seed"])):
            writer.writerow(row)

    print(f"Summary written to {summary_path}")
    print(f"Total runs: {len(summary_rows)}")

    # Compute config-level means and stds
    print("\n── Config Summary (mean ± std over 3 seeds) ──")
    print(f"{'Config':<12} {'Score':<16} {'Loss':<16} {'Survival':<16} {'Entropy':<12} {'SharkDist':<12}")
    print("-" * 80)
    for key in sorted(all_data.keys()):
        scores = [d["final_score"] for d in all_data[key]]
        losses = [d["final_loss"] for d in all_data[key]]
        survivals = [d["mean_survival"] for d in all_data[key]]
        entropies = [d["action_entropy"] for d in all_data[key]]
        dists = [d["mean_shark_dist"] for d in all_data[key]]
        print(f"{key:<12} {np.mean(scores):>6.1f}±{np.std(scores):>4.1f}  "
              f"{np.mean(losses):>7.3f}±{np.std(losses):>5.3f}  "
              f"{np.mean(survivals):>7.1f}±{np.std(survivals):>4.1f}  "
              f"{np.mean(entropies):>5.2f}±{np.std(entropies):>3.2f}  "
              f"{np.mean(dists):>6.1f}±{np.std(dists):>4.1f}")

    plot_learning_curves(all_data)
    print(f"\nPlots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()