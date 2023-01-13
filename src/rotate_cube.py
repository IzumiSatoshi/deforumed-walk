import pygame
import numpy as np
from math import *


WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

WIDTH, HEIGHT = 899, 600
pygame.display.set_caption("3D projection in pygame!")
screen = pygame.display.set_mode((WIDTH, HEIGHT))


def resize_and_centerlize(point):
    point *= 500  # resize
    point[0] += WIDTH / 2  # centerlize
    point[1] += HEIGHT / 2  # centerlize
    return point


# x, y, z, 1
# fmt: off
points = np.array([
    [3, 1, -1],
    [-3, 1, -1],
    [-3, -1, -1],
    [3, -1, -1],
    [3, 1, 1],
    [-3, 1, 1],
    [-3, -1, 1],
    [3, -1, 1],
])




clock = pygame.time.Clock()
t = 0
while True:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                exit()

    # update stuff
    screen.fill(WHITE)

    rotation_x_matrix = ([
        [1, 0, 0],
        [0, cos(radians(t)), -sin(radians(t))],
        [0, sin(radians(t)), cos(radians(t))],
    ])

    rotation_y_matrix = ([
        [cos(radians(t)), 0, sin(radians(t))],
        [0, 1, 0],
        [-sin(radians(t)), 0, cos(radians(t))],
    ])
    rotation_z_matrix = ([
        [cos(radians(t)), -sin(radians(t)), 0],
        [sin(radians(t)), cos(radians(t)), 0],
        [0, 0, 1],
    ])


    point_2d_list = []
    for point in points:
        point = rotation_x_matrix @ point
        point = rotation_y_matrix @ point
        point = rotation_z_matrix @ point

        projection_matrix = ([
            [0, 1/(5 - point[0]), 0],
            [0, 0, 1/(5 - point[0])],
        ])

        point_2d = projection_matrix @ point
        point_2d = resize_and_centerlize(point_2d)
        point_2d_list.append(point_2d)
        pygame.draw.circle(screen, BLACK, (point_2d[0], point_2d[1]), 5)
    
    pygame.draw.line(screen, BLACK, point_2d_list[0], point_2d_list[1])
    pygame.draw.line(screen, BLACK, point_2d_list[1], point_2d_list[2])
    pygame.draw.line(screen, BLACK, point_2d_list[2], point_2d_list[3])
    pygame.draw.line(screen, BLACK, point_2d_list[0], point_2d_list[3])
    pygame.draw.line(screen, BLACK, point_2d_list[0+4], point_2d_list[1+4])
    pygame.draw.line(screen, BLACK, point_2d_list[1+4], point_2d_list[2+4])
    pygame.draw.line(screen, BLACK, point_2d_list[2+4], point_2d_list[3+4])
    pygame.draw.line(screen, BLACK, point_2d_list[0+4], point_2d_list[3+4])
    pygame.draw.line(screen, BLACK, point_2d_list[0], point_2d_list[0+4])
    pygame.draw.line(screen, BLACK, point_2d_list[1], point_2d_list[1+4])
    pygame.draw.line(screen, BLACK, point_2d_list[2], point_2d_list[2+4])
    pygame.draw.line(screen, BLACK, point_2d_list[3], point_2d_list[3+4])

    pygame.display.update()

    t += 1
