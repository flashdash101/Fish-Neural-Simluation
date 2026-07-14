# Predator-Prey Simulation: Multi-Agent DQN

A 2D predator-prey reinforcement learning environment where schools of fish learn to evade a curriculum-driven shark predator using Deep Q-Networks (DQN) with ray-casting perception.

Inspired by the **BioInspired Optimisation** module studied at university -- exploring emergent collective behaviour under distributed predation risk -- this project investigates whether learned avoidance policies improve with larger school sizes and whether ray-count (perception acuity) interacts with school size.

## Results

| Config | School | Rays | Score | Survival (frames) | Loss |
|--------|--------|------|-------|-------------------|------|
| S1R4   | 1      | 4    | 254 +- 69 | 2,910 | 0.90 |
| S1R8   | 1      | 8    | 355 +- 100 | 3,374 | 0.92 |
| S2R8   | 2      | 8    | 236 | 3,331 | 2.39 |
| S4R4   | 4      | 4    | **378** | **7,535** | 1.95 |
| S4R8   | 4      | 8    | 283 | 3,270 | 12.07 |

Key findings:
- **School size dominated ray count.** S4R4 achieved the highest score (378) and longest survival (7,535 frames), outperforming single-fish baselines by 48%.
- **Distributed predation risk stabilised training.** Single-fish agents faced life-or-death at every timestep; schools of 4 benefited from the shark only targeting one fish at a time, providing natural "breaks" that regularised Q-value estimates.
- **Ray count showed an interaction effect.** At school=1, more rays helped (S1R8 > S1R4). At school=4, more rays did not improve over 4-ray baselines, suggesting diminishing returns when schooling already provides sufficient threat information.
- **School=2 exposed a stability boundary.** Bimodal shark-targeting dynamics (two fish, frequent switching) triggered Q-value overestimation collapse in one seed; this was diagnosed and resolved via gradient clipping and learning-rate tuning.
- **Shark distance was flat across configs (~300--375 px).** Fish did not win by staying far from the predator; they won by surviving longer, highlighting the survival-time advantage of schooling.

### Evolution: Fixed vs Random Spawn

| | Fixed Spawn (early) | Random Spawn (final) |
|---|---|---|
| Shark position | Bottom-left corner only | Anywhere in arena |
| Learned behaviour | Memoised "top-right corner" | Directional evasion |
| Generalisation | Poor | Robust |

The fixed-spawn agent learned to hide in the corner opposite the shark's static position. Randomising the shark spawn forced the policy to learn genuine directional evasion -- flee away from wherever the threat appears.

## Implementation

### Environment
- 800x800 px arena with 18 px border margin
- Fish school of 1--4 agents sharing a single DQN via parameter sharing
- One shark predator with curriculum speed (8.5 px/s at episode 0, cap 13.0 px/s at episode 500)
- Random spawn positions for both predator and prey (episode-reset)

### Perception: Ray-Casting Engine
- Custom 2D ray-segment intersection engine built from scratch
- Configurable fan of rays (FOV = pi/2, 4--8 rays) projected from each fish's heading
- Segments include: arena walls, shark body (8 segments), other fish bodies (8 segments each)
- Rendered every 3rd frame during training to reduce rendering overhead

### Optimisations
- Vectorised ray-segment intersection using NumPy broadcasting (replaced O(rays x segments) Python double-loop) -- 3--5x speedup
- Precomputed static box-wall segments -- eliminated redundant allocations in the hot `get_state()` path
- Headless training mode with uncapped simulation loop and 10x environment steps per rendered frame
- CUDA auto-detection for DQN forward/backward passes
- Optional Numba JIT-compiled ray function (experimental; vectorised NumPy version used in final runs)

### DQN Agent
- Architecture: 3-layer MLP (state_dim -> 128 -> 128 -> 5 actions)
- Replay buffer: 100k experiences, batch size 64
- Epsilon-greedy exploration: decay 0.999/step, min 0.01
- Gradient clipping: max_norm=1.0
- Soft target network updates: tau=1e-3
- NaN/Inf loss guarding
- GPU auto-detection via `torch.cuda.is_available()`

### Reward Function
- Shark avoidance: reward increasing distance from shark (delta * 0.3)
- Wall penalty: smooth penalty within 40 px of walls ((40 - dist) * 0.03)
- Obstacle penalty: smooth penalty from close ray hits ((20 - min_ray) * 0.03)
- Survival bonus: +0.15 per step
- Total reward clipped to [-10.0, 2.0]

### Training Pipeline
- Ablation study: 6 configurations x 2 seeds x 650 episodes
- Google Colab GPU runtime with automatic Drive sync for persistence
- Per-episode CSVs: score, loss, survival time, action distribution, mean shark distance
- Best-weight checkpointing per configuration

## Demo Video

```
python demo.py --school 2 --rays 8
```

Loads trained weights and runs the simulation with a pure greedy policy (epsilon=0), showing the learned evasion behaviour live. Supports frame export for video creation:

```
python demo.py --school 2 --rays 8 --save-frames --episodes 3
```

### Video Comparisons

| | Fixed Spawn (Old) | Random Spawn (New) |
|---|---|---|
| **Video** | *[insert old video link]* | *[insert new video link]* |
| Shark position | Bottom-left corner | Randomised each episode |
| Fish behaviour | Hugs top-right corner | Flees dynamically from threat |
| Ray count | 8 | 8 |
| School size | 2 | 2 |

## Results Gallery

| Description | Figure |
|---|---|
| Final scores by configuration | *[insert final_score_bars.png]* |
| School x Rays score heatmap | *[insert score_heatmap.png]* |
| Learning curves (mean +- std) | *[insert learning_curves.png]* |
| Survival time and shark distance | *[insert survival_sharkdist.png]* |
| Action distribution heatmap | *[insert action_heatmap.png]* |
| Correlation: survival vs score | *[insert correlations.png]* |

### Old vs New Results

| | Fixed Spawn (Old) | Random Spawn (New) |
|---|---|---|
| S1R4 Score | *[insert old S1R4 plot]* | *[insert new S1R4 plot]* |
| S2R4 Score | *[insert old S2R4 plot]* | *[insert new S2R4 plot]* |
| Learning curves | *[insert old learning curves]* | *[insert new learning curves]* |

## Setup

### Requirements

```
pygame-ce>=2.5.0
numpy>=1.24.0
torch>=2.0.0
matplotlib>=3.7.0
pandas>=1.5.0
scipy>=1.10.0
numba>=0.57.0
```

Install:

```
pip install -r requirements.txt
```

### Run Training Locally

```
python run_ablation.py
```

Edit `CONFIGS` and `SEEDS` inside `run_ablation.py` to control which experiments run.

### Google Colab

1. Upload `Initial_copy.py` (rename to `Initial.py`), `model.py`, and `run_ablation.py`
2. Mount Google Drive for persistent results
3. Set `USE_DRIVE = True` in `run_ablation.py`
4. Run `!python run_ablation.py`

Completed runs auto-skip on re-run (resume logic).

### Analyse Results

```
python collect_results.py     # generates results/summary.csv and learning curves
python analyse_results.py     # generates 8 comprehensive analysis figures
```

Outputs appear in `results/analysis/`.

## Future Aspirations

### Dual DQN: Fish vs Shark
Replace the hand-crafted shark tracking policy with a second DQN agent. Both agents learn simultaneously in a competitive self-play setup -- the fish learns evasion while the shark learns pursuit. This would produce emergent predator-prey co-evolution and naturally remove the need for a curriculum shark speed.

### Genetic Algorithms
Evolve neural network architectures and hyperparameters (learning rate, gamma, epsilon decay) across a population of agents, selecting for survival time and score stability. Combined with the ablation framework, this could automatically discover optimal configurations for each school size.

### Multi-Shark Environment
Introduce multiple predator agents to increase environmental complexity and test whether schooling benefits scale under heightened threat density.

## Project Structure

```
fish-neural-simulation/
  Initial.py             # Original simulation (fixed-spawn, for demo compatibility)
  Initial_copy.py        # Main simulation (random spawn, vectorised rays, 650 episodes)
  model.py               # DQN agent, replay buffer, Q-network (GPU-ready)
  run_ablation.py        # Ablation study runner with Drive sync
  collect_results.py     # Quick summary CSV + learning curves
  analyse_results.py     # Full analysis: 8 figures + per-config aggregates
  demo.py                # Live demo using trained weights
  results/
    raw/                 # Per-episode CSVs from ablation runs
    weights/             # Best-weight checkpoints (.pth)
    analysis/            # Generated figures and analysis_summary.csv
    plots/               # Learning curve plots
```

## License

MIT
