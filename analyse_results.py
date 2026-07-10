"""
analyze_results.py
Expanded analysis of the ablation study. Reads every per-run CSV in
results/raw/ and produces:
  1. results/analysis/final_score_bars.png      - grouped bar w/ error bars (TOP 1)
  2. results/analysis/score_heatmap.png         - school x rays heatmap   (TOP 2)
  3. results/analysis/learning_curves.png       - mean +/- std per config
  4. results/analysis/action_heatmap.png        - action usage per config
  5. results/analysis/survival_sharkdist.png    - survival time & shark dist bars
  6. results/analysis/sample_efficiency.png     - episodes to 90% of final score
  7. results/analysis/correlations.png          - survival/sharkdist vs score scatter
  8. results/analysis/analysis_summary.csv      - per-config aggregates

Handles partial / single-seed data: std and tests only computed where n >= 2.
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "results", "raw")
OUT_DIR = os.path.join(BASE_DIR, "results", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

NUM_ACTIONS = 5
# Action index -> human-readable label (must match apply_action in Initial.py)
ACTION_LABELS = ["turn left", "turn right", "up", "down", "speed up"]


def load_run(filepath):
    """Load one CSV into a dict of episode-level arrays."""
    scores, losses, lengths, dists = [], [], [], []
    actions = np.zeros((0, NUM_ACTIONS), dtype=int)
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row["score"]))
            if row["loss"]:
                losses.append(float(row["loss"]))
            lengths.append(int(row["ep_length"]))
            dists.append(float(row["shark_dist_avg"]))
            actions = np.vstack([actions, [int(row[f"action_{i}"]) for i in range(NUM_ACTIONS)]])
    return {
        "scores": np.array(scores),
        "losses": np.array(losses),
        "lengths": np.array(lengths),
        "dists": np.array(dists),
        "actions": actions,
    }


def last100_mean(a):
    n = len(a)
    if n == 0:
        return np.nan
    sl = slice(max(0, n - 100), n)
    return float(np.mean(a[sl]))


def main():
    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".csv"))
    if not files:
        print("No CSVs in results/raw/. Run run_ablation.py first.")
        return

    # Group runs by config label
    configs = {}
    for fname in files:
        parts = fname.replace(".csv", "").split("_")
        school = int(parts[0].replace("school", ""))
        rays = int(parts[1].replace("rays", ""))
        seed = int(parts[2].replace("seed", ""))
        label = f"S{school}R{rays}"
        data = load_run(os.path.join(RAW_DIR, fname))
        data["school"], data["rays"], data["seed"] = school, rays, seed
        configs.setdefault(label, []).append(data)

    # Per-config aggregates
    agg = {}
    for label, runs in configs.items():
        scores = [last100_mean(r["scores"]) for r in runs]
        surv = [last100_mean(r["lengths"]) for r in runs]
        dist = [last100_mean(r["dists"]) for r in runs]
        loss = [last100_mean(r["losses"]) for r in runs]
        # action distribution pooled over runs (last-100 episodes each)
        act_pool = np.zeros(NUM_ACTIONS, dtype=int)
        for r in runs:
            n = len(r["actions"])
            sl = slice(max(0, n - 100), n)
            act_pool += r["actions"][sl].sum(axis=0)
        agg[label] = {
            "school": runs[0]["school"],
            "rays": runs[0]["rays"],
            "n": len(runs),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)) if len(scores) > 1 else 0.0,
            "surv_mean": float(np.mean(surv)),
            "surv_std": float(np.std(surv)) if len(surv) > 1 else 0.0,
            "dist_mean": float(np.mean(dist)),
            "dist_std": float(np.std(dist)) if len(dist) > 1 else 0.0,
            "loss_mean": float(np.mean(loss)),
            "actions": act_pool,
        }

    labels = sorted(agg.keys(), key=lambda k: (agg[k]["school"], agg[k]["rays"]))

    # ── 1. Final-score grouped bar with error bars (TOP 1) ──────────
    fig, ax = plt.subplots(figsize=(9, 5))
    means = [agg[l]["score_mean"] for l in labels]
    errs = [agg[l]["score_std"] for l in labels]
    bars = ax.bar(labels, means, yerr=errs, capsize=5,
                  color=plt.cm.viridis(np.linspace(0.2, 0.9, len(labels))))
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.0f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Final avg score (last 100 eps)")
    ax.set_title("Final Score by Config (mean ± std)\n"
                 "label = S<school>R<rays>  (e.g. S2R8 = school of 2, 8 rays)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "final_score_bars.png"), dpi=150)
    plt.close(fig)

    # ── 2. School x Rays heatmap (TOP 2) ───────────────────────────
    schools = sorted({agg[l]["school"] for l in labels})
    rays = sorted({agg[l]["rays"] for l in labels})
    grid = np.full((len(schools), len(rays)), np.nan)
    for l in labels:
        i = schools.index(agg[l]["school"])
        j = rays.index(agg[l]["rays"])
        grid[i, j] = agg[l]["score_mean"]
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(rays)))
    ax.set_xticklabels(rays)
    ax.set_yticks(range(len(schools)))
    ax.set_yticklabels(schools)
    ax.set_xlabel("Ray count")
    ax.set_ylabel("School size")
    ax.set_title("Final Score Heatmap (school × rays)")
    for i in range(len(schools)):
        for j in range(len(rays)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center",
                        color="white", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, label="score")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "score_heatmap.png"), dpi=150)
    plt.close(fig)

    # ── 3. Learning curves mean +/- std ────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for l in labels:
        runs = configs[l]
        arrs = [r["scores"] for r in runs]
        min_len = min(len(a) for a in arrs)
        trunc = np.array([a[:min_len] for a in arrs])
        m = trunc.mean(axis=0)
        s = trunc.std(axis=0)
        x = np.arange(1, min_len + 1)
        ax.plot(x, m, label=f"{l} (n={len(runs)})")
        ax.fill_between(x, m - s, m + s, alpha=0.15)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title("Learning Curves (mean ± std)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "learning_curves.png"), dpi=150)
    plt.close(fig)

    # ── 4. Action-distribution heatmap ─────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    act_grid = np.array([agg[l]["actions"] for l in labels], dtype=float)
    act_grid /= act_grid.sum(axis=1, keepdims=True)  # row-normalise
    im = ax.imshow(act_grid, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_ACTIONS))
    ax.set_xticklabels([f"a{i}\n{ACTION_LABELS[i]}" for i in range(NUM_ACTIONS)])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Action")
    ax.set_ylabel("Config")
    ax.set_title("Action Usage Distribution (last 100 eps, normalised)\n"
                 "a0=turn left  a1=turn right  a2=up  a3=down  a4=speed up")
    for i in range(len(labels)):
        for j in range(NUM_ACTIONS):
            ax.text(j, i, f"{act_grid[i, j]:.2f}", ha="center", va="center",
                    color="black" if act_grid[i, j] < 0.5 else "white", fontsize=8)
    fig.colorbar(im, ax=ax, label="fraction")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "action_heatmap.png"), dpi=150)
    plt.close(fig)

    # ── 5. Survival time & shark distance bars ─────────────────────
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    surv = [agg[l]["surv_mean"] for l in labels]
    surv_e = [agg[l]["surv_std"] for l in labels]
    a1.bar(labels, surv, yerr=surv_e, capsize=4, color="teal")
    a1.set_ylabel("Avg survival (frames)")
    a1.set_title("Survival Time by Config")
    a1.tick_params(axis="x", rotation=45)
    a1.grid(axis="y", alpha=0.3)
    dist = [agg[l]["dist_mean"] for l in labels]
    dist_e = [agg[l]["dist_std"] for l in labels]
    a2.bar(labels, dist, yerr=dist_e, capsize=4, color="orange")
    a2.set_ylabel("Avg shark distance (px)")
    a2.set_title("Mean Shark Distance by Config")
    a2.tick_params(axis="x", rotation=45)
    a2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "survival_sharkdist.png"), dpi=150)
    plt.close(fig)

    # ── 6. Sample efficiency: episodes to sustained 90% of final ──
    # Smooth the mean curve (window 25) and require the curve to stay
    # above 90% of the final value (sustained), so early noise spikes
    # don't count as "converged".
    fig, ax = plt.subplots(figsize=(9, 4.5))
    eff = []
    for l in labels:
        runs = configs[l]
        arrs = [r["scores"] for r in runs]
        min_len = min(len(a) for a in arrs)
        m = np.array([a[:min_len] for a in arrs]).mean(axis=0)
        # moving-average smoothing
        w = 25
        if len(m) >= w:
            kernel = np.ones(w) / w
            sm = np.convolve(m, kernel, mode="same")
            # trim edge artefacts
            sm[:w] = m[:w]
            sm[-w:] = m[-w:]
        else:
            sm = m
        target = 0.9 * sm[-1]
        above = sm >= target
        # find first index after which it stays above for the rest
        idx = np.where(above)[0]
        if len(idx) and above[idx[0]:].all():
            eff.append(int(idx[0]) + 1)
        else:
            eff.append(min_len)  # never sustained -> full length
    ax.bar(labels, eff, color="purple")
    for i, e in enumerate(eff):
        ax.text(i, e, f"{e}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Episodes to sustained 90% of final")
    ax.set_title("Sample Efficiency by Config\n"
                 "(fewer episodes = faster convergence; 500 = not yet converged)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "sample_efficiency.png"), dpi=150)
    plt.close(fig)

    # ── 7. Correlations: survival & shark dist vs score ───────────
    # Pool per-episode points across all runs for a global view.
    all_surv, all_dist, all_score = [], [], []
    for l in labels:
        for r in configs[l]:
            all_surv.append(r["lengths"])
            all_dist.append(r["dists"])
            all_score.append(r["scores"])
    all_surv = np.concatenate(all_surv)
    all_dist = np.concatenate(all_dist)
    all_score = np.concatenate(all_score)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.scatter(all_surv, all_score, s=4, alpha=0.2, color="teal")
    r1, p1 = stats.pearsonr(all_surv, all_score)
    a1.set_xlabel("Survival (frames)")
    a1.set_ylabel("Score")
    a1.set_title(f"Survival vs Score (r={r1:.2f}, p={p1:.1e})")
    a1.grid(alpha=0.3)
    a2.scatter(all_dist, all_score, s=4, alpha=0.2, color="orange")
    r2, p2 = stats.pearsonr(all_dist, all_score)
    a2.set_xlabel("Shark distance (px)")
    a2.set_ylabel("Score")
    a2.set_title(f"Shark Dist vs Score (r={r2:.2f}, p={p2:.1e})")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "correlations.png"), dpi=150)
    plt.close(fig)

    # ── 8. analysis_summary.csv ───────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "analysis_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "school", "rays", "n_seeds",
                    "score_mean", "score_std", "surv_mean", "surv_std",
                    "dist_mean", "dist_std", "loss_mean",
                    "act_0", "act_1", "act_2", "act_3", "act_4"])
        for l in labels:
            a = agg[l]
            w.writerow([l, a["school"], a["rays"], a["n"],
                        round(a["score_mean"], 2), round(a["score_std"], 2),
                        round(a["surv_mean"], 1), round(a["surv_std"], 1),
                        round(a["dist_mean"], 2), round(a["dist_std"], 2),
                        round(a["loss_mean"], 4),
                        *[int(x) for x in a["actions"]]])

    # ── Console report ────────────────────────────────────────────
    print(f"Analyzed {len(files)} runs across {len(labels)} configs.")
    print(f"Correlation survival~score: r={r1:.3f} (p={p1:.1e})")
    print(f"Correlation sharkdist~score: r={r2:.3f} (p={p2:.1e})")
    print("\n── Per-config (last-100-ep mean) ──")
    print(f"{'cfg':<7}{'n':>3}{'score':>9}{'surv':>9}{'sharkD':>9}{'loss':>8}")
    for l in labels:
        a = agg[l]
        print(f"{l:<7}{a['n']:>3}{a['score_mean']:>9.1f}"
              f"{a['surv_mean']:>9.0f}{a['dist_mean']:>9.1f}{a['loss_mean']:>8.3f}")
    print(f"\nFigures + analysis_summary.csv written to {OUT_DIR}")


if __name__ == "__main__":
    main()
