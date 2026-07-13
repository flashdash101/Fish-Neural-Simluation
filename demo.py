"""
demo.py
Run a trained policy for a quick demo / video capture.

Loads saved weights from results/weights/ and plays the simulation with a
PURE GREEDY policy (epsilon forced to 0) so the behaviour shown is the
learned policy, not random exploration.

Usage:
  # Play live (record with OBS / Xbox Game Bar):
  python demo.py --school 2 --rays 8

  # Save frames as PNGs to results/demo_frames/ for a video:
  python demo.py --school 2 --rays 8 --save-frames --episodes 3

Then turn frames into a video (ffmpeg):
  ffmpeg -framerate 60 -i results/demo_frames/frame_%06d.png -c:v libx264 \
         -pix_fmt yuv420p results/demo.mp4
"""

import os
import sys
import math
import random
import argparse

import numpy as np
import pygame
import torch

# Reuse the environment + rendering from Initial.py
from Initial import (
    WIDTH, HEIGHT, BG_COLOR, BOX_COLOR, BOX_MARGIN,
    get_state, step, shark_track, nearest_alive_fish,
    draw_shark, draw_fish,
    segments_from_box, segments_from_shark, segments_from_other_fish,
    generate_rays_fan, raycast_rays_segments,
    closest_intersection_from_raycast_rays_segments,
)
from model import DQNAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "results", "weights")
FRAMES_DIR = os.path.join(BASE_DIR, "results", "demo_frames")


class Fish:
    """Minimal fish matching the one used in Initial.main()."""
    def __init__(self, x, y, heading):
        self.x = x
        self.y = y
        self.heading = heading
        self.alive = True
        self.return_value = 0


def find_weights(school, rays, seed=None):
    """Locate the best-weight file for a given config."""
    if seed is not None:
        p = os.path.join(WEIGHTS_DIR, f"school{school}_rays{rays}_seed{seed}_best.pth")
        return p if os.path.exists(p) else None
    # pick the first matching seed if no seed specified
    for s in [42, 43, 44]:
        p = os.path.join(WEIGHTS_DIR, f"school{school}_rays{rays}_seed{s}_best.pth")
        if os.path.exists(p):
            return p
    return None


def run_demo(school=2, rays=8, seed=None, episodes=3, save_frames=False,
             fps=60, shark_episode=499, steps_per_frame=5):
    weights_path = find_weights(school, rays, seed)
    if weights_path:
        print(f"Using weights: {weights_path}")
    if weights_path is None:
        print(f"No weights found for school={school} rays={rays}. "
              f"Run the ablation first or train with Initial.py.")
        sys.exit(1)
    print(f"Loading weights: {weights_path}")

    action_space = 5
    state_space = 7 + rays
    agent = DQNAgent(state_size=state_space, action_size=action_space)
    # Load weights — map to CPU so Colab-trained (GPU) weights work locally
    state_dict = torch.load(weights_path, weights_only=True, map_location=torch.device("cpu"))
    agent.qnetwork_local.load_state_dict(state_dict)
    agent.qnetwork_target.load_state_dict(state_dict)
    #Use a non-greedy policy for the demo, so the fish will always take the best action according to the Q network
    agent.epsilon = 0.9
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Fish Demo — school={school} rays={rays} (trained)")
    clock = pygame.time.Clock()

    if save_frames:
        os.makedirs(FRAMES_DIR, exist_ok=True)

    shark_x = BOX_MARGIN + 120
    shark_y = HEIGHT - BOX_MARGIN - 120
    shark_heading = -math.pi / 4
    shark_vx = math.cos(shark_heading) * 25.0
    shark_vy = math.sin(shark_heading) * 25.0

    school_fish = [
        Fish(random.randint(BOX_MARGIN, WIDTH - BOX_MARGIN),
             random.randint(BOX_MARGIN, HEIGHT - BOX_MARGIN),
             random.uniform(0, 2 * math.pi))
        for _ in range(school)
    ]

    running = True
    episode_count = 0
    frame_count = 0
    saved = 0

    while running and episode_count < episodes:
        dt = min(clock.tick(fps) / 1000.0, 1.0 / 30.0)

        # Multiple environment steps per frame (matches training cadence),
        # so the fish must actually react to the shark rather than idling.
        for _ in range(steps_per_frame):
            for fish in school_fish:
                if not fish.alive:
                    continue
                state = get_state(fish.x, fish.y, fish.heading, shark_x, shark_y,
                                  school_fish, rays, self_fish=fish)
                action = agent.greedy_action(state)
                next_state, reward, done, (fish.x, fish.y, fish.heading) = step(
                    state, action, fish.x, fish.y, fish.heading,
                    shark_x, shark_y, school_fish, rays, self_fish=fish)
                if done:
                    fish.alive = False

            target = nearest_alive_fish(school_fish, shark_x, shark_y)
            if target is not None:
                # Use the TRAINED shark speed (end-of-training curriculum),
                # not the slow starting speed, so the policy is challenged.
                shark_vx, shark_vy = shark_track(target.x, target.y, shark_x, shark_y,
                                                shark_vx, shark_vy, shark_episode)
                shark_x += shark_vx * dt
                shark_y += shark_vy * dt
                shark_heading = math.atan2(shark_vy, shark_vx)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if all(not f.alive for f in school_fish):
            episode_count += 1
            print(f"Episode {episode_count} ended.")
            school_fish = [
                Fish(random.randint(BOX_MARGIN, WIDTH - BOX_MARGIN),
                     random.randint(BOX_MARGIN, HEIGHT - BOX_MARGIN),
                     random.uniform(0, 2 * math.pi))
                for _ in range(school)
            ]
            shark_x = BOX_MARGIN + 120
            shark_y = HEIGHT - BOX_MARGIN - 120
            shark_heading = -math.pi / 4
            shark_vx = math.cos(shark_heading) * 25.0
            shark_vy = math.sin(shark_heading) * 25.0

        # ── Render ──
        screen.fill(BG_COLOR)
        pygame.draw.rect(
            screen, BOX_COLOR,
            pygame.Rect(BOX_MARGIN, BOX_MARGIN,
                        WIDTH - BOX_MARGIN * 2, HEIGHT - BOX_MARGIN * 2), 3)
        draw_shark(screen, (shark_x, shark_y), shark_heading)
        for fish in school_fish:
            if not fish.alive:
                continue
            draw_fish(screen, (fish.x, fish.y), fish.heading)
            fish_segments = segments_from_box(
                (BOX_MARGIN, BOX_MARGIN),
                (WIDTH - BOX_MARGIN, HEIGHT - BOX_MARGIN))
            fish_segments.extend(segments_from_shark((shark_x, shark_y), shark_heading))
            for other in school_fish:
                if not other.alive or other is fish:
                    continue
                fish_segments.extend(segments_from_other_fish((other.x, other.y), other.heading))
            fish_segments = np.array(fish_segments, dtype=float)
            rays_fan = generate_rays_fan((fish.x, fish.y), fish.heading,
                                        fov=math.pi / 2, num=rays)
            intersections = raycast_rays_segments(rays_fan, fish_segments)
            closest = closest_intersection_from_raycast_rays_segments(intersections)
            for inter in closest:
                pygame.draw.line(screen, (255, 0, 0),
                                 (int(fish.x), int(fish.y)),
                                 (int(inter[0]), int(inter[1])), 1)

        pygame.display.flip()

        if save_frames:
            pygame.image.save(screen, os.path.join(FRAMES_DIR, f"frame_{saved:06d}.png"))
            saved += 1

        frame_count += 1

    pygame.quit()
    if save_frames:
        print(f"Saved {saved} frames to {FRAMES_DIR}")
        print("To make a video:\n"
              f"  ffmpeg -framerate {fps} -i {FRAMES_DIR}/frame_%06d.png "
              "-c:v libx264 -pix_fmt yuv420p results/demo.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo a trained fish policy.")
    parser.add_argument("--school", type=int, default=2)
    parser.add_argument("--rays", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--shark-episode", type=int, default=499,
                        help="Curriculum episode used for shark speed "
                             "(higher = faster shark, matching trained policy).")
    parser.add_argument("--steps-per-frame", type=int, default=5,
                        help="Environment steps per rendered frame "
                             "(matches training cadence).")
    parser.add_argument("--save-frames", action="store_true")
    args = parser.parse_args()
    run_demo(school=args.school, rays=args.rays, seed=args.seed,
             episodes=args.episodes, save_frames=args.save_frames, fps=args.fps,
             shark_episode=args.shark_episode, steps_per_frame=args.steps_per_frame)
