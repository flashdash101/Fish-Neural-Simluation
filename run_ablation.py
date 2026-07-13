"""
run_ablation.py
Runs the ablation study over school sizes and ray counts.
Saves per-episode data to CSV and best weights to .pth for each run.

"""

import csv
import os
import random
import numpy as np
import torch
import shutil
from Initial_copy import main  # our modified main()

# ── Google Drive sync (Colab only) ──────────────────────────
# Set to True to redirect results to Google Drive for persistence.
USE_DRIVE = False  # change to True in Colab before running

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if USE_DRIVE:
    # Verify Drive is actually mounted (not just a local path)
    if not os.path.exists("/content/drive/MyDrive"):
        print("ERROR: Drive not mounted! Run this in a Colab cell first:")
        print("    from google.colab import drive")
        print("    drive.mount('/content/drive')")
        print("Falling back to local results/ folder instead.")
        USE_DRIVE = False
        RAW_DIR = os.path.join(BASE_DIR, "results", "raw")
        WEIGHTS_DIR = os.path.join(BASE_DIR, "results", "weights")
    else:
        DRIVE_BASE = "/content/drive/MyDrive/fish_simulation_results"
        RAW_DIR = os.path.join(DRIVE_BASE, "raw")
        WEIGHTS_DIR = os.path.join(DRIVE_BASE, "weights")
        print(f"Drive mode ON — results will persist at {DRIVE_BASE}/")
else:
    RAW_DIR = os.path.join(BASE_DIR, "results", "raw")
    WEIGHTS_DIR = os.path.join(BASE_DIR, "results", "weights")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ── Configurations ───────────────────────────────────────────
# n configs × k seeds = n * k runs.
CONFIGS = [
    # (2, 4),
    (2,8)      # re-run: only 500 rows, needs full 650
]
# k seeds per config for testing; use [42,43,44] for full ablation
SEEDS = [44]  


def entropy(counts):
    """Compute the entropy of a dict of action counts."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counts.values()])
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def run_config(school_size, ray_count, seed):
    """Run one config and save its data."""
    print(f"── [{seed}] school={school_size}  rays={ray_count} ──")
    data = main(
        num_fish=school_size,
        num_rays=ray_count,
        headless=True,
        seed=seed,
    )

    # Build per-episode CSV
    csv_path = os.path.join(RAW_DIR, f"school{school_size}_rays{ray_count}_seed{seed}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "score", "loss", "ep_length",
            "action_0", "action_1", "action_2", "action_3", "action_4",
            "shark_dist_avg",
        ])
        for ep in range(len(data["episode_scores"])):
            ah = data["action_histories"][ep]
            writer.writerow([
                ep + 1,
                round(data["episode_scores"][ep], 4),
                round(data["episode_losses"][ep], 6) if ep < len(data["episode_losses"]) else "",
                data["episode_lengths"][ep],
                ah.get(0, 0), ah.get(1, 0), ah.get(2, 0), ah.get(3, 0), ah.get(4, 0),
                round(data["shark_dist_avgs"][ep], 2) if ep < len(data["shark_dist_avgs"]) else "",
            ])

    # Save best weights — copy to both local results/ AND Drive (if enabled).
    best_weights_src = os.path.join(BASE_DIR, "best_weights.pth")
    if os.path.exists(best_weights_src):
        # Always save a local copy
        local_dst = os.path.join(BASE_DIR, "results", "weights",
                                 f"school{school_size}_rays{ray_count}_seed{seed}_best.pth")
        os.makedirs(os.path.dirname(local_dst), exist_ok=True)
        shutil.copy2(best_weights_src, local_dst)
        print(f"  Saved weights to {local_dst}")
        # If Drive is enabled, also save there (shutil.copy2 handles cross-device)
        if USE_DRIVE:
            drive_dst = os.path.join(WEIGHTS_DIR,
                                     f"school{school_size}_rays{ray_count}_seed{seed}_best.pth")
            os.makedirs(os.path.dirname(drive_dst), exist_ok=True)
            shutil.copy2(best_weights_src, drive_dst)
            print(f"  Saved weights to Drive: {drive_dst}")

    print(f"  Done – {len(data['episode_scores'])} episodes")
    return data


def run_ablation(skip_existing=True):
    print("=" * 60)
    print("Ablation Study Runner")
    print(f"Configs: {len(CONFIGS)} × seeds {SEEDS} = {len(CONFIGS) * len(SEEDS)} runs")
    if skip_existing:
        print("Resume mode: runs with an existing CSV will be skipped.")
    print("=" * 60)

    for school_size, ray_count in CONFIGS:
        for seed in SEEDS:
            csv_path = os.path.join(
                RAW_DIR, f"school{school_size}_rays{ray_count}_seed{seed}.csv"
            )
            if skip_existing and os.path.exists(csv_path):
                print(f"── SKIP [{seed}] school={school_size} rays={ray_count} "
                      f"(already done) ──")
                continue
            try:
                run_config(school_size, ray_count, seed)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n All runs complete. Results in {RAW_DIR}/")


if __name__ == "__main__":
    run_ablation()