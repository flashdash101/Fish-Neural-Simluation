import math
import sys

import pygame
import numpy as np
from random import randint

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


def generate_rays_fan(view_position, direction, fov, num=20):
    angles = np.linspace(-fov/2, fov/2, num=num) + direction

    rays = np.empty((angles.shape[0], 2, 2))
    rays[:, 0, 0] = view_position[0]
    rays[:, 0, 1] = view_position[1]
    rays[:, 1, 0] = view_position[0] + np.cos(angles)
    rays[:, 1, 1] = view_position[1] + np.sin(angles)
    return rays 

def unique_points_from_segments(segments):
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

def segments_from_shark(center, angle):
    x1, y1 = center
    body_length = 150
    body_height = 60
    tail_length = 50

    body_points = [
        (x1 - body_length * 0.5, y1),
        (x1 - body_length * 0.25, y1 + body_height * 0.5),
        (x1 + body_length * 0.28, y1 + body_height * 0.43),
        (x1 + body_length * 0.5, y1),
        (x1 + body_length * 0.28, y1 - body_height * 0.43),
        (x1 - body_length * 0.25, y1 - body_height * 0.5),
    ]
    segments = []
    for i in range(len(body_points)):
        p1 = body_points[i]
        p2 = body_points[(i + 1) % len(body_points)]
        segments.append((p1, p2))

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


def main():
    pygame.init()
    shark_x = BOX_MARGIN + 120
    shark_y = HEIGHT - BOX_MARGIN - 120
    shark_speed = 140.0
    shark_heading = -math.pi / 4
    shark_vx = math.cos(shark_heading) * shark_speed
    shark_vy = math.sin(shark_heading) * shark_speed
    shark_turn_timer = 0.0
   
    segments = []
    #Include shark and fish in the segments list
    segments += segments_from_box((BOX_MARGIN, BOX_MARGIN), (WIDTH - BOX_MARGIN, HEIGHT - BOX_MARGIN))
    segments += segments_from_shark((shark_x, shark_y), shark_heading)
    segments = np.array(segments)
    points = unique_points_from_segments(segments)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Fish Simulation")
    clock = pygame.time.Clock()

    orbit_center = (WIDTH // 2, HEIGHT // 2)
    orbit_radius = 140
    orbit_speed = 1.2
    motion_angle = 0.0

   




    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        dt = clock.tick(6000) / 1000.0
        motion_angle += orbit_speed * dt

        fish_x = orbit_center[0] + math.cos(motion_angle) * orbit_radius
        fish_y = orbit_center[1] + math.sin(motion_angle) * orbit_radius
        fish_heading = motion_angle + math.pi / 2

        shark_turn_timer -= dt
        if shark_turn_timer <= 0:
            shark_heading = randint(0, 628) / 100.0
            shark_speed = randint(80, 180)
            shark_vx = math.cos(shark_heading) * shark_speed
            shark_vy = math.sin(shark_heading) * shark_speed
            shark_turn_timer = randint(40, 140) / 100.0

        shark_x += shark_vx * dt
        shark_y += shark_vy * dt

        if shark_x < BOX_MARGIN + 40:
            shark_x = BOX_MARGIN + 40
            shark_vx = abs(shark_vx)
            shark_turn_timer = 0
        elif shark_x > WIDTH - BOX_MARGIN - 40:
            shark_x = WIDTH - BOX_MARGIN - 40
            shark_vx = -abs(shark_vx)
            shark_turn_timer = 0

        if shark_y < BOX_MARGIN + 25:
            shark_y = BOX_MARGIN + 25
            shark_vy = abs(shark_vy)
            shark_turn_timer = 0
        elif shark_y > HEIGHT - BOX_MARGIN - 25:
            shark_y = HEIGHT - BOX_MARGIN - 25
            shark_vy = -abs(shark_vy)
            shark_turn_timer = 0

        shark_heading = math.atan2(shark_vy, shark_vx)

            

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
        # Draw visible rays from the fish to the closest intersections on the box
        for intersection in closest_intersections:
            pygame.draw.line(
                screen,
                (255, 0, 0),
                (int(fish_x), int(fish_y)),
                (int(intersection[0]), int(intersection[1])),
                1,
            )
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
