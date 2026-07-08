import math
import sys

import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from model import DQNAgent, ReplayBuffer, QNetwork
from collections import deque
WIDTH, HEIGHT = 800, 600
BG_COLOR = (18, 24, 38)
FISH_COLOR = (240, 170, 70)
FIN_COLOR = (220, 120, 40)
EYE_COLOR = (20, 20, 20)
BOX_COLOR = (235, 235, 235)
BOX_MARGIN = 18


SHARK_COLOR = (200, 200, 200)
SHARK_FIN_COLOR = (150, 150, 150)

"""
ray is a 2D ray defined by two points: the origin and a point along the ray's direction.
segment is a 2D line segment defined by its two endpoints.
returns the intersection point of the ray and the segment if they intersect, otherwise returns None.


"""



        


def raycast_ray_segment(ray, segment):
    r_px = ray[0,0]
    r_py = ray[0,1]
    r_dx = ray[1,0] - ray[0,0]
    r_dy = ray[1,1] - ray[0,1]

    s_px = segment[0,0]
    s_py = segment[0,1]
    s_dx = segment[1,0] - segment[0,0]
    s_dy = segment[1,1] - segment[0,1]

    r_mag = math.sqrt(r_dx * r_dx + r_dy * r_dy)
    s_mag = math.sqrt(s_dx * s_dx + s_dy * s_dy)
                      
    if r_dx / r_mag == s_dx / s_mag and r_dy / r_mag == s_dy / s_mag:
        return None

    try:
        T2 = (r_dx * (s_py - r_py) + r_dy * (r_px - s_px)) / (s_dx * r_dy - s_dy * r_dx)
    except ZeroDivisionError:
        return None

    try:
        T1 = (s_px + s_dx * T2 - r_px) / r_dx
    except ZeroDivisionError:
        T1 = (s_py + s_dy * T2 - r_py) / (r_dx - 0.0001)
		

    if T1 < 0:
        return None
    if T2 < 0 or T2 > 1: return None


    return (r_px + r_dx * T1, r_py + r_dy * T1, T1)

def generate_rays_from_points(view_position, points):
    angles = np.arctan2(points[:, 1] - view_position[1], points[:, 0] - view_position[0])
    # sort angles for correct polygon recontruction
    angles = np.flip(np.sort(np.concatenate((angles - 0.00001, angles + 0.00001))), 0)

    """ return unit vectors pointing to each point in the scene ...
        once the amount of points becomes very high, will become more
        efficent to just create equally spaced rays around the look position """
    rays = np.empty((angles.shape[0], 2, 2))
    rays[:, 0, 0] = view_position[0]
    rays[:, 0, 1] = view_position[1]
    rays[:, 1, 0] = view_position[0] + np.cos(angles)
    rays[:, 1, 1] = view_position[1] + np.sin(angles)

    return rays

def generate_rays_circle(view_position, num=100):
    angles = np.linspace(0, 2*np.pi, num=num)

    rays = np.empty((angles.shape[0], 2, 2))
    rays[:, 0, 0] = view_position[0]
    rays[:, 0, 1] = view_position[1]
    rays[:, 1, 0] = view_position[0] + np.cos(angles)
    rays[:, 1, 1] = view_position[1] + np.sin(angles)
    return rays

import time
#Function to calculate the relative distance between two points in 2D space
def calculate_relative_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

#Calculate the change in distance between two points in 2D space over a small time interval
def calculate_relative_newdistance(x1, y1, x2, y2):
    old_distance = calculate_relative_distance(x1, y1, x2, y2)
    time.sleep(0.1)
    new_distance = calculate_relative_distance(x1, y1, x2, y2)
    return new_distance - old_distance

#Function to calculate the relvative angle between two points in 2D space
def calculate_relative_angle(x1, y1, x2, y2):
    return np.arctan2(y2 - y1, x2 - x1)

def generate_rays_fan(view_position, direction, fov, num=8):
    angles = np.linspace(-fov/2, fov/2, num=num) + direction

    rays = np.empty((angles.shape[0], 2, 2))
    rays[:, 0, 0] = view_position[0]
    rays[:, 0, 1] = view_position[1]
    rays[:, 1, 0] = view_position[0] + np.cos(angles)
    rays[:, 1, 1] = view_position[1] + np.sin(angles)
    return rays 

def unique_points_from_segments(segments):
    segments = np.array(segments, dtype=float)
    all_points = segments.reshape(-1, 2)
    return np.unique(all_points, axis=0)

def raycast_rays_segments(rays, segments):
    intersections = np.full((rays.shape[0], segments.shape[0], 3), np.nan, dtype=float)

    for i, ray in enumerate(rays):
        for j, segment in enumerate(segments):
            hit = raycast_ray_segment(ray, segment)
            if hit is not None:
                intersections[i, j] = hit

    return intersections

def closest_intersection_from_raycast_rays_segments(intersections):
    # get closest intersections (rays, segments, (x,y,T1))

    # remove rays with no intersection (full nan return on the final axis, causes nanargmin to throw error)
    # kinda obscure code
    n = (~np.isnan(intersections).any(axis=-1)).any(axis=-1)
    intersections = intersections[n,:,:]

    closest = np.nanargmin(intersections[:, :, 2], axis=1)
    return intersections[list(range(0, intersections.shape[0])), closest, :2]

def segments_from_box(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    s = []
    s.append([[x1,y1],[x2,y1]])
    s.append([[x2,y1],[x2,y2]])
    s.append([[x2,y2],[x1,y2]])
    s.append([[x1,y2],[x1,y1]])
    return s

#Create actions to turn left, turn right, up, down and speed up etc.
def create_actions():
    actions = []
    actions.append(0) #turn left
    actions.append(1) #turn right
    actions.append(2) #up
    actions.append(3) #down
    actions.append(4) #speed up
    return actions


#Define the segments that make up the shark shape based on its center and angle
def segments_from_shark(center, angle):
   x,y = center
   body_length = 150
   body_height = 60
   tail_length = 50

   direction = (math.cos(angle), math.sin(angle))
   perpendicular = (-direction[1], direction[0])

   def point(forward, side):
       return (
           x + direction[0] * forward + perpendicular[0] * side,
           y + direction[1] * forward + perpendicular[1] * side,
       )
   
   body_points = [
         point(-body_length * 0.5, 0),
          point(-body_length * 0.25, body_height * 0.5),
          point(body_length * 0.28, body_height * 0.43),
          point(body_length * 0.5, 0),
          point(body_length * 0.28, -body_height * 0.43),
          point(-body_length * 0.25, -body_height * 0.5),
     ]
   
   tail_base = point(-body_length * 0.5, 0)
   tail_top = point(-body_length * 0.5 - tail_length, tail_length * 0.65)
   tail_bottom = point(-body_length * 0.5 - tail_length, -tail_length * 0.65)
   
   segments = []
   for i in range(len(body_points)):
        segments.append([body_points[i], body_points[(i + 1) % len(body_points)]])
   segments.append([tail_base, tail_top])
   segments.append([tail_base, tail_bottom])


   return segments


def segements_from_other_fish(center, angle):
       x,y = center
       body_length = 90
       body_height = 40
       tail_length = 28

       direction = (math.cos(angle), math.sin(angle))
       perpendicular = (-direction[1], direction[0])

       def point(forward, side):
           return (
               x + direction[0] * forward + perpendicular[0] * side,
               y + direction[1] * forward + perpendicular[1] * side,
           )
       
       body_points = [
             point(-body_length * 0.5, 0),
              point(-body_length * 0.25, body_height * 0.5),
              point(body_length * 0.28, body_height * 0.43),
              point(body_length * 0.5, 0),
              point(body_length * 0.28, -body_height * 0.43),
              point(-body_length * 0.25, -body_height * 0.5),
         ]
       
       tail_base = point(-body_length * 0.5, 0)
       tail_top = point(-body_length * 0.5 - tail_length, tail_length * 0.65)
       tail_bottom = point(-body_length * 0.5 - tail_length, -tail_length * 0.65)
       
       segments = []
       for i in range(len(body_points)):
            segments.append([body_points[i], body_points[(i + 1) % len(body_points)]])
       segments.append([tail_base, tail_top])
       segments.append([tail_base, tail_bottom])


       return segments




def draw_segments(screen, segments):
    for p1, p2 in segments:
        pygame.draw.line(screen, (0,0,0), p1, p2, 1)


    

def draw_shark(surface, center, angle):
    x, y = center
    body_length = 150
    body_height = 60
    tail_length = 50

    direction = (math.cos(angle), math.sin(angle))
    perpendicular = (-direction[1], direction[0])
    def point(forward, side):
        return (
            x + direction[0] * forward + perpendicular[0] * side,
            y + direction[1] * forward + perpendicular[1] * side,
        )
    body_points = [
        point(-body_length * 0.5, 0),
        point(-body_length * 0.25, body_height * 0.5),
        point(body_length * 0.28, body_height * 0.43),
        point(body_length * 0.5, 0),
        point(body_length * 0.28, -body_height * 0.43),
        point(-body_length * 0.25, -body_height * 0.5),
    ]

    tail_base = point(-body_length * 0.5, 0)
    tail_top = point(-body_length * 0.5 - tail_length, tail_length
    * 0.65)
    tail_bottom = point(-body_length * 0.5 - tail_length, -tail_length * 0.65)

    fin_top = [
        point(-body_length * 0.12, -body_height * 0.2),
        point(0, -body_height * 0.95),
        point(body_length * 0.08, -body_height * 0.15),
    ]

    fin_bottom = [
        point(-body_length * 0.05, body_height * 0.15),
        point(body_length * 0.12, body_height * 0.85),
        point(body_length * 0.22, body_height * 0.12),
    ]

    pygame.draw.polygon(surface, SHARK_COLOR, body_points)
    pygame.draw.polygon(surface, SHARK_COLOR, [tail_base, tail_top, tail_bottom])
    pygame.draw.polygon(surface, SHARK_FIN_COLOR, fin_top)
    pygame.draw.polygon(surface, SHARK_FIN_COLOR, fin_bottom)

    eye_position = point(body_length * 0.18, -body_height * 0.14)
    pygame.draw.circle(surface, EYE_COLOR, (int(eye_position[0]), int(eye_position[1])), 4)



#Probably not efficient, but it works for now. Draws the fish shape based on its center and angle

def draw_fish(surface, center, angle):
    x, y = center

    body_length = 90
    body_height = 40
    tail_length = 28

    direction = (math.cos(angle), math.sin(angle))
    perpendicular = (-direction[1], direction[0])

    def point(forward, side):
        return (
            x + direction[0] * forward + perpendicular[0] * side,
            y + direction[1] * forward + perpendicular[1] * side,
        )

    body_points = [
        point(-body_length * 0.5, 0),
        point(-body_length * 0.25, body_height * 0.5),
        point(body_length * 0.28, body_height * 0.43),
        point(body_length * 0.5, 0),
        point(body_length * 0.28, -body_height * 0.43),
        point(-body_length * 0.25, -body_height * 0.5),
    ]

    tail_base = point(-body_length * 0.5, 0)
    tail_top = point(-body_length * 0.5 - tail_length, tail_length * 0.65)
    tail_bottom = point(-body_length * 0.5 - tail_length, -tail_length * 0.65)

    fin_top = [
        point(-body_length * 0.12, -body_height * 0.2),
        point(0, -body_height * 0.95),
        point(body_length * 0.08, -body_height * 0.15),
    ]

    fin_bottom = [
        point(-body_length * 0.05, body_height * 0.15),
        point(body_length * 0.12, body_height * 0.85),
        point(body_length * 0.22, body_height * 0.12),
    ]

    pygame.draw.polygon(surface, FISH_COLOR, body_points)
    pygame.draw.polygon(surface, FISH_COLOR, [tail_base, tail_top, tail_bottom])
    pygame.draw.polygon(surface, FIN_COLOR, fin_top)
    pygame.draw.polygon(surface, FIN_COLOR, fin_bottom)

    eye_position = point(body_length * 0.18, -body_height * 0.14)
    pygame.draw.circle(surface, EYE_COLOR, (int(eye_position[0]), int(eye_position[1])), 4)
    

def calculate_relative_distance_from_bounds(fish_x, fish_y, box_margin, width, height):
    distance_from_top_wall = fish_y - box_margin
    distance_from_bottom_wall = height - box_margin - fish_y
    distance_from_left_wall = fish_x - box_margin
    distance_from_right_wall = width - box_margin - fish_x

    return distance_from_top_wall, distance_from_bottom_wall, distance_from_left_wall, distance_from_right_wall   


def segments_from_other_fish(center, angle):
    x,y = center
    body_length = 90
    body_height = 40
    tail_length = 28

    direction = (math.cos(angle), math.sin(angle))
    perpendicular = (-direction[1], direction[0])

    def point(forward, side):
        return (
            x + direction[0] * forward + perpendicular[0] * side,
            y + direction[1] * forward + perpendicular[1] * side,
        )
    
    body_points = [
          point(-body_length * 0.5, 0),
           point(-body_length * 0.25, body_height * 0.5),
           point(body_length * 0.28, body_height * 0.43),
           point(body_length * 0.5, 0),
           point(body_length * 0.28, -body_height * 0.43),
           point(-body_length * 0.25, -body_height * 0.5),
      ]
    
    tail_base = point(-body_length * 0.5, 0)
    tail_top = point(-body_length * 0.5 - tail_length, tail_length * 0.65)
    tail_bottom = point(-body_length * 0.5 - tail_length, -tail_length * 0.65)
    
    segments = []
    for i in range(len(body_points)):
         segments.append([body_points[i], body_points[(i + 1) % len(body_points)]])
    segments.append([tail_base, tail_top])
    segments.append([tail_base, tail_bottom])


    return segments
#In main we willcreate the environment and run the simulation, we will also create the DQN agent and the replay buffer, and use them to train the fish to avoid the shark
def get_state(fish_x, fish_y, fish_heading, shark_x, shark_y, school, self_fish=None):
    # Primary threat signal: the shark
    distance = calculate_relative_distance(fish_x, fish_y, shark_x, shark_y)
    angle = calculate_relative_angle(fish_x, fish_y, shark_x, shark_y) - fish_heading

    distance_from_top_wall, distance_from_bottom_wall, distance_from_left_wall, distance_from_right_wall = calculate_relative_distance_from_bounds(fish_x, fish_y, BOX_MARGIN, WIDTH, HEIGHT)
    velocity = 0  # We will add velocity later

    # Rays are ALWAYS computed from this fish's own position.
    # Other fish are added as obstacles (excluding self) so the fish can sense them.
    rays = generate_rays_fan((fish_x, fish_y), fish_heading, fov=math.pi / 2, num=8)
    segments = segments_from_box((BOX_MARGIN, BOX_MARGIN), (WIDTH - BOX_MARGIN, HEIGHT - BOX_MARGIN))
    shark_segments = segments_from_shark((shark_x, shark_y), calculate_relative_angle(fish_x, fish_y, shark_x, shark_y))
    segments.extend(shark_segments)
    for fish in school:
        if not fish.alive:
            continue
        if self_fish is not None and fish is self_fish:
            continue
        fish_segments = segments_from_other_fish((fish.x, fish.y), fish.heading)
        segments.extend(fish_segments)
    segments = np.array(segments, dtype=float)

    intersections = raycast_rays_segments(rays, segments)
    closest_intersections = closest_intersection_from_raycast_rays_segments(intersections)
    ray_distances = np.linalg.norm(closest_intersections - np.array([fish_x, fish_y]), axis=1)
    # Pad ray_distances to ensure consistent size (8 rays)
    if len(ray_distances) < 8:
        ray_distances = np.concatenate([ray_distances, np.full(8 - len(ray_distances), np.inf)])
                                                            
    return np.array([distance, angle, velocity, distance_from_top_wall, distance_from_bottom_wall, distance_from_left_wall, distance_from_right_wall, *ray_distances], dtype=np.float32)




def apply_action(fish_x, fish_y, fish_heading, action):
    fish_speed = 2.0
    if action == 0: #turn left
        fish_heading -= 0.1
    elif action == 1: #turn right
        fish_heading += 0.1
    elif action == 2: #up
        fish_y -= fish_speed
    elif action == 3: #down
        fish_y += fish_speed
    elif action == 4: #speed up
        fish_x += fish_speed * math.cos(fish_heading)
        fish_y += fish_speed * math.sin(fish_heading)
    return fish_x, fish_y, fish_heading

def check_collision(fish_x, fish_y, shark_x, shark_y):
    distance = calculate_relative_distance(fish_x, fish_y, shark_x, shark_y)
    distance_from_top_wall, distance_from_bottom_wall, distance_from_left_wall, distance_from_right_wall = calculate_relative_distance_from_bounds(fish_x, fish_y, BOX_MARGIN, WIDTH, HEIGHT)
    if distance < 50: #If the fish is too close to the shark, it dies and the episode ends
        return True
    if distance_from_top_wall < 0 or distance_from_bottom_wall < 0 or distance_from_left_wall < 0 or distance_from_right_wall < 0:
        return True

    return False

#Reward the fish based on its distance to the shark, the closer it is, the worse the reward
#Give a small reward for surving each step, and a large negative reward for dying
#Give an extra reward every 250 frames for surviving
#Upon death, reset the fish to the center of the screen and reset the shark to a random position
#Penalise the fish for going out of bounds, and reward it for staying within the box
#Add smoothing reward so that the fish is penalised for getting too clouse to shark and to walls, and rewarded for staying away from them
def compute_reward(fish_x, fish_y, shark_x, shark_y, prev_distance, ray_distances):
    distance = calculate_relative_distance(fish_x, fish_y, shark_x, shark_y)

    # Handle NaN/inf in ray distances
    ray_distances = np.array(ray_distances)
    ray_distances = ray_distances[np.isfinite(ray_distances)]
    min_ray = np.min(ray_distances) if len(ray_distances) > 0 else float('inf')

    # Minimum distance to any of the 4 walls
    wall_distance = min(fish_x - BOX_MARGIN, WIDTH - BOX_MARGIN - fish_x,
                        fish_y - BOX_MARGIN, HEIGHT - BOX_MARGIN - fish_y)

    # Terminal events: caught by shark or fully out of bounds
    if distance < 50:
        return -10.0
    if wall_distance < 0:
        return -10.0

    # Smooth wall penalty: only kicks in within 40px of a wall, gentle slope.
    # This gives gradual warning without penalising most of the box.
    wall_penalty = 0.0
    if wall_distance < 40:
        wall_penalty = (40 - wall_distance) * 0.02  # max ~0.8 at the wall

    # Smooth obstacle penalty from rays (shark body / other fish)
    obstacle_penalty = 0.0
    if min_ray < 20:
        obstacle_penalty = (20 - min_ray) * 0.03  # max ~0.6

    # Shark avoidance: reward INCREASING distance from the shark (delta).
    # This is the main positive learning signal; scaled up so fleeing pays off.
    delta = distance - prev_distance
    shark_reward = delta * 0.3

    # Small survival bonus (kept modest so avoidance dominates learning).
    survival = 0.15

    reward = shark_reward - wall_penalty - obstacle_penalty + survival
    # Clip total reward to keep Q-values bounded and training stable.
    reward = max(-10.0, min(2.0, reward))
    return reward

#A function for the shark to track the fish, stochastically, not too fast.
#We will add some inertia to the shark's movement
#We will make the shark choose a target fish position to track, and then move towards that position with some randomness
def shark_track(fish_x, fish_y, shark_x, shark_y, shark_vx, shark_vy):
    # Calculate the direction to the fish
    direction_to_fish = np.array([fish_x - shark_x, fish_y - shark_y])
    # Normalize the direction vector
    distance_to_fish = np.linalg.norm(direction_to_fish)
    #Find the minimumu distance to any fish in the school, and track that fish instead of the first fish in the list
    if distance_to_fish > 0:
        direction_to_fish = direction_to_fish / distance_to_fish
        # Update the shark's velocity based on the direction to the fish, with some randomness
        shark_speed = 7.0
        randomness = np.random.uniform(-0.1, 0.1, size=2)  # Randomness in the direction
        shark_vx = direction_to_fish[0] * shark_speed + randomness[0]
        shark_vy = direction_to_fish[1] * shark_speed + randomness[1]
    return shark_vx, shark_vy

#A strong normalised first state
def reset_positions():
    fish_x = WIDTH // 2
    fish_y = HEIGHT // 2
    fish_heading = 0

    shark_x = BOX_MARGIN + 120
    shark_y = HEIGHT - BOX_MARGIN - 120
    shark_speed = 7.0
    shark_heading = -math.pi / 4
    shark_vx = math.cos(shark_heading) * shark_speed
    shark_vy = math.sin(shark_heading) * shark_speed

    return fish_x, fish_y, fish_heading, shark_x, shark_y, shark_vx, shark_vy


#Plot progress from 0 to n episodes, with a moving average of 10 episodes for easier reading.
def plot_scores(episode_scores, episode_losses):
    import matplotlib.pyplot as plt


    episodes = np.arange(1, len(episode_scores) + 1)
    # Smooth rewards with a moving average for easier reading
    window = 10
    if len(episode_scores) >= window:
        reward_ma = np.convolve(episode_scores, np.ones(window) / window, mode='valid')
        reward_x = np.arange(window, len(episode_scores) + 1)
    else:
        reward_ma = episode_scores
        reward_x = episodes
    plt.figure(figsize=(12, 5))


    plt.plot(episodes, episode_scores, alpha=0.35, label='Reward per Episode')
    plt.plot(reward_x, reward_ma, linewidth=2, label=f'{window}-Episode Moving Average')
    plt.title('Training Progress: Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_rewards.png', dpi=150, bbox_inches='tight')
    plt.close()


    if len(episode_losses) >= window:
        # episode_losses may be longer than episode_scores (appended per step).
        # Align by taking the tail so x/y dimensions match for plotting.
        n_scores = len(episode_scores)
        losses_plot = episode_losses[-n_scores:] if len(episode_losses) >= n_scores else episode_losses
        loss_ma = np.convolve(losses_plot, np.ones(window) / window, mode='valid')
        loss_x = np.arange(window, len(losses_plot) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(1, len(losses_plot) + 1), losses_plot, alpha=0.35, label='Loss per Episode')
        plt.plot(loss_x, loss_ma, linewidth=2, label=f'{window}-Episode Moving Average')
        plt.title('Training Progress: Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('training_losses.png', dpi=150, bbox_inches='tight')
        plt.close()




def step(state, action, fish_x, fish_y, fish_heading, shark_x, shark_y, school, self_fish=None):
    # Update the fish's position and heading based on the action taken
    fish_speed = 3.0
    if action == 0: #turn left
        fish_heading -= 0.1
    elif action == 1: #turn right
        fish_heading += 0.1
    elif action == 2: #up
        fish_y -= fish_speed
    elif action == 3: #down
        fish_y += fish_speed
    elif action == 4: #speed up
        fish_x += fish_speed * math.cos(fish_heading)
        fish_y += fish_speed * math.sin(fish_heading)
    

    # Calculate the new state and reward
    new_state = get_state(fish_x, fish_y, fish_heading, shark_x, shark_y, school, self_fish)
    
    prev_distance = state[0]  # The previous distance to the shark is the first element of the state
    distance = new_state[0]
    ray_distances = new_state[7:]  # The new ray distances are the elements from index 7 onwards
    reward = compute_reward(fish_x, fish_y, shark_x, shark_y, prev_distance, ray_distances)

    # Penalise being too close to other fish so they spread out and avoid collisions
    for fish in school:
        if not fish.alive:
            continue
        if self_fish is not None and fish is self_fish:
            continue
        fish_distance = calculate_relative_distance(fish_x, fish_y, fish.x, fish.y)
        if fish_distance < 30:
            reward -= 0.1

    done = check_collision(fish_x, fish_y, shark_x, shark_y)

    return new_state, reward, done, (fish_x, fish_y, fish_heading)


import random
from random import randint

#Return the nearest alive fish to the shark, or None if the school is empty
def nearest_alive_fish(school, shark_x, shark_y):
    best = None
    best_dist = float('inf')
    for fish in school:
        if not fish.alive:
            continue
        d = calculate_relative_distance(shark_x, shark_y, fish.x, fish.y)
        if d < best_dist:
            best_dist = d
            best = fish
    return best

#Need to make training faster, so we will run multiple environmental updates per frame
#Multiple training steps per frame, and multiple environment updates per frame, to speed up training
def main():
    action_space = 5
    state_space = 15  # 7 base features + 8 ray distances
    agent = DQNAgent(state_size=state_space, action_size=action_space)

    scores_window = deque(maxlen=100)
    episode_scores = []
    episode_losses = []

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Fish Simulation")
    clock = pygame.time.Clock()

    shark_x = BOX_MARGIN + 120
    shark_y = HEIGHT - BOX_MARGIN - 120
    shark_heading = -math.pi / 4
    shark_vx = math.cos(shark_heading) * 25.0
    shark_vy = math.sin(shark_heading) * 25.0

    running = True
    episode_return = 0
    episode_count = 0
    episode_loss_sum = 0.0
    episode_loss_steps = 0
    frame_count = 0
   
    class Fish:
        def __init__(self, x, y, heading):
            self.x = x
            self.y = y
            self.heading = heading
            self.alive = True
            self.return_value = 0

    school = [Fish(randint(BOX_MARGIN, WIDTH - BOX_MARGIN), randint(BOX_MARGIN, HEIGHT - BOX_MARGIN), random.uniform(0, 2 * math.pi)) for _ in range(3)]

    while running:
        dt = min(clock.tick(60) / 1000.0, 1.0 / 30.0)  # cap dt so the shark can't teleport on slow frames
        #Every frame, we will update the environment multiple times to speed up training
        for _ in range(5):  # Update the environment 5 times per frame
            for fish in school:
                if not fish.alive:
                    continue
                state = get_state(fish.x, fish.y, fish.heading, shark_x, shark_y, school, self_fish=fish)
                action = agent.greedy_action(state)
                next_state, reward, done, (fish.x, fish.y, fish.heading) = step(state, action, fish.x, fish.y, fish.heading, shark_x, shark_y, school, self_fish=fish)
                episode_return += reward
                agent.memory.add(state, action, reward, next_state, done)
                #Learn every 3rd frame
                if len(agent.memory) > 64 and frame_count % 3 == 0:
                    loss = agent.learn()
                else:
                    loss = None

                if loss is not None:
                    episode_loss_sum += loss
                    episode_loss_steps += 1
                if done:
                    fish.alive = False



            # Shark chases the nearest alive fish (not the old ghost fish)
            target = nearest_alive_fish(school, shark_x, shark_y)
            if target is not None:
                shark_vx, shark_vy = shark_track(target.x, target.y, shark_x, shark_y, shark_vx, shark_vy)
                shark_x += shark_vx * dt
                shark_y += shark_vy * dt
                shark_heading = math.atan2(shark_vy, shark_vx)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Only reset the whole school when EVERY fish is dead
            if all(not fish.alive for fish in school):
                episode_count += 1
                scores_window.append(episode_return)
                # Append ONE loss value per episode (mean over the episode's steps)
                if episode_loss_steps > 0:
                    episode_losses.append(episode_loss_sum / episode_loss_steps)
                episode_loss_sum = 0.0
                episode_loss_steps = 0
                if episode_count > 0 and episode_count % 50 == 0:
                    avg_score = np.mean(scores_window) if len(scores_window) > 0 else 0
                    avg_loss = episode_losses[-1] if episode_losses else 0
                    print(f'Episode {episode_count}: Avg Score={avg_score:.2f}, Loss={avg_loss:.4f}, Epsilon={agent.epsilon:.4f}')

                if episode_count >= 500 and episode_count % 500 == 0:
                    plot_scores(episode_scores, episode_losses)
                agent.decay_epsilon()
                episode_return = 0
                school = [Fish(randint(BOX_MARGIN, WIDTH - BOX_MARGIN), randint(BOX_MARGIN, HEIGHT - BOX_MARGIN), random.uniform(0, 2 * math.pi)) for _ in range(3)]
                shark_x = BOX_MARGIN + 120
                shark_y = HEIGHT - BOX_MARGIN - 120
                shark_heading = -math.pi / 4
                shark_vx = math.cos(shark_heading) * 25.0
                shark_vy = math.sin(shark_heading) * 25.0

      
        


        screen.fill(BG_COLOR)
        pygame.draw.rect(
            screen,
            BOX_COLOR,
            pygame.Rect(BOX_MARGIN, BOX_MARGIN, WIDTH - BOX_MARGIN * 2, HEIGHT - BOX_MARGIN * 2),
            3,
        )

        draw_shark(screen, (shark_x, shark_y), shark_heading)
        for fish in school:
            if fish.alive:
                draw_fish(screen, (fish.x, fish.y), fish.heading)
                # Build a per-fish segment list that excludes the fish itself.
                fish_segments = segments_from_box((BOX_MARGIN, BOX_MARGIN), (WIDTH - BOX_MARGIN, HEIGHT - BOX_MARGIN))
                shark_segments = segments_from_shark((shark_x, shark_y), shark_heading)
                fish_segments.extend(shark_segments)
                for other_fish in school:
                    if not other_fish.alive or other_fish is fish:
                        continue
                    other_segments = segments_from_other_fish((other_fish.x, other_fish.y), other_fish.heading)
                    fish_segments.extend(other_segments)
                fish_segments = np.array(fish_segments, dtype=float)

                # Draw THIS fish's rays from its own position.
                #Draw the rays for the fish every 3rd frame, to speed up training and reduce rendering load
                if frame_count % 3 == 0:
                    rays = generate_rays_fan((fish.x, fish.y), fish.heading, fov=math.pi / 2, num=8)
                    intersections = raycast_rays_segments(rays, fish_segments)
                    closest_intersections = closest_intersection_from_raycast_rays_segments(intersections)
                    for intersection in closest_intersections:
                        pygame.draw.line(
                            screen,
                            (255, 0, 0),
                            (int(fish.x), int(fish.y)),
                            (int(intersection[0]), int(intersection[1])),
                            1,
                        )

        frame_count += 1

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
    
