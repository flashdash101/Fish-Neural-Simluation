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


#Function to calculate the relative distance between two points in 2D space
def calculate_relative_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

#Function to calculate the relvative angle between two points in 2D space
def calculate_relative_angle(x1, y1, x2, y2):
    return np.arctan2(y2 - y1, x2 - x1)

def generate_rays_fan(view_position, direction, fov, num=20):
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
#In main we willcreate the environment and run the simulation, we will also create the DQN agent and the replay buffer, and use them to train the fish to avoid the shark
def get_state(fish_x, fish_y, fish_heading, shark_x, shark_y):
    distance = calculate_relative_distance(fish_x, fish_y, shark_x, shark_y)
    distance_from_top_wall, distance_from_bottom_wall, distance_from_left_wall, distance_from_right_wall = calculate_relative_distance_from_bounds(fish_x, fish_y, BOX_MARGIN, WIDTH, HEIGHT)
    angle = calculate_relative_angle(fish_x, fish_y, shark_x, shark_y) - fish_heading
    velocity = 0 #We will add velocity later
    return np.array([distance, angle, velocity, distance_from_top_wall, distance_from_bottom_wall, distance_from_left_wall, distance_from_right_wall], dtype=np.float32)

def apply_action(fish_x, fish_y, fish_heading, action):
    if action == 0: #turn left
        fish_heading -= 0.1
    elif action == 1: #turn right
        fish_heading += 0.1
    elif action == 2: #up
        fish_y -= 5
    elif action == 3: #down
        fish_y += 5
    elif action == 4: #speed up
        fish_x += 5 * math.cos(fish_heading)
        fish_y += 5 * math.sin(fish_heading)
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
def compute_reward(fish_x, fish_y, shark_x, shark_y):
    distance = calculate_relative_distance(fish_x, fish_y, shark_x, shark_y)
    if distance < 50:
        return -1000
    reward = -distance

    distance_from_top_wall, distance_from_bottom_wall, distance_from_left_wall, distance_from_right_wall = calculate_relative_distance_from_bounds(fish_x, fish_y, BOX_MARGIN, WIDTH, HEIGHT)
    if distance_from_top_wall < 0 or distance_from_bottom_wall < 0 or distance_from_left_wall < 0 or distance_from_right_wall < 0:
        reward -= 100
    else:
        reward += 10
    #Now for every frame that the fish survives, give it a small reward of 1
    reward += 1
    #Now for every 250 frames that the fish survives, give it an extra reward of 100
    if pygame.time.get_ticks() % 250 == 0:
        reward += 100

    #Return the reward
    return reward

#A strong normalised first state
def reset_positions():
    fish_x = WIDTH // 2
    fish_y = HEIGHT // 2
    fish_heading = 0

    shark_x = BOX_MARGIN + 120
    shark_y = HEIGHT - BOX_MARGIN - 120
    shark_speed = 140.0
    shark_heading = -math.pi / 4
    shark_vx = math.cos(shark_heading) * shark_speed
    shark_vy = math.sin(shark_heading) * shark_speed

    return fish_x, fish_y, fish_heading, shark_x, shark_y, shark_vx, shark_vy

def step(state, action, fish_x, fish_y, fish_heading, shark_x, shark_y):
    #Update the fish's position and heading based on the action taken
    if action == 0: #turn left
        fish_heading -= 0.1
    elif action == 1: #turn right
        fish_heading += 0.1
    elif action == 2: #up
        fish_y -= 5
    elif action == 3: #down
        fish_y += 5
    elif action == 4: #speed up
        fish_x += 5 * math.cos(fish_heading)
        fish_y += 5 * math.sin(fish_heading)

    #Calculate the new state and reward
    new_state = get_state(fish_x, fish_y, fish_heading, shark_x, shark_y)
    distance = new_state[0]
    reward = compute_reward(fish_x, fish_y, shark_x, shark_y)

    done = False
    if distance < 50: #If the fish is too close to the shark, it dies and the episode ends
        done = True

    return new_state, reward, done, (fish_x, fish_y, fish_heading)

def main():
    action_space = 5
    state_space = 7
    agent = DQNAgent(state_size=state_space, action_size=action_space)

    scores_window = deque(maxlen=100)
    episode_scores = []
    episode_losses = []

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Fish Simulation")
    clock = pygame.time.Clock()

    fish_x, fish_y, fish_heading, shark_x, shark_y, shark_vx, shark_vy = reset_positions()
    running = True

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        state = get_state(fish_x, fish_y, fish_heading, shark_x, shark_y)
        action = agent.greedy_action(state)
        fish_x, fish_y, fish_heading = apply_action(fish_x, fish_y, fish_heading, action)

        shark_x += shark_vx * dt
        shark_y += shark_vy * dt

        if shark_x < BOX_MARGIN + 40:
            shark_x = BOX_MARGIN + 40
            shark_vx = abs(shark_vx)
        elif shark_x > WIDTH - BOX_MARGIN - 40:
            shark_x = WIDTH - BOX_MARGIN - 40
            shark_vx = -abs(shark_vx)

        if shark_y < BOX_MARGIN + 25:
            shark_y = BOX_MARGIN + 25
            shark_vy = abs(shark_vy)
        elif shark_y > HEIGHT - BOX_MARGIN - 25:
            shark_y = HEIGHT - BOX_MARGIN - 25
            shark_vy = -abs(shark_vy)

        shark_heading = math.atan2(shark_vy, shark_vx)

        reward = compute_reward(fish_x, fish_y, shark_x, shark_y)
        next_state = get_state(fish_x, fish_y, fish_heading, shark_x, shark_y)
        done = check_collision(fish_x, fish_y, shark_x, shark_y)

        agent.memory.add(state, action, reward, next_state, done)
        loss = agent.learn()
        agent.decay_epsilon()

        scores_window.append(reward)
        episode_scores.append(reward)
        if loss is not None:
            episode_losses.append(loss)

        segments = segments_from_box((BOX_MARGIN, BOX_MARGIN), (WIDTH - BOX_MARGIN, HEIGHT - BOX_MARGIN))
        shark_segments = segments_from_shark((shark_x, shark_y), shark_heading)
        segments.extend(shark_segments)
        segments = np.array(segments, dtype=float)

        rays = generate_rays_fan((fish_x, fish_y), fish_heading, fov=math.pi / 2, num=20)
        intersections = raycast_rays_segments(rays, segments)
        closest_intersections = closest_intersection_from_raycast_rays_segments(intersections)

        screen.fill(BG_COLOR)
        pygame.draw.rect(
            screen,
            BOX_COLOR,
            pygame.Rect(BOX_MARGIN, BOX_MARGIN, WIDTH - BOX_MARGIN * 2, HEIGHT - BOX_MARGIN * 2),
            3,
        )

        draw_shark(screen, (shark_x, shark_y), shark_heading)
        draw_fish(screen, (fish_x, fish_y), fish_heading)

        for intersection in closest_intersections:
            pygame.draw.line(
                screen,
                (255, 0, 0),
                (int(fish_x), int(fish_y)),
                (int(intersection[0]), int(intersection[1])),
                1,
            )

        pygame.display.flip()

        if done:
            fish_x, fish_y, fish_heading, shark_x, shark_y, shark_vx, shark_vy = reset_positions()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
