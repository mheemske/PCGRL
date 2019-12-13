import pygame
import numpy as np
import math
import random
import itertools
import tensorflow as tf

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREY = (126, 126, 126)
 
# Set the width and height of each square
SQUARE_WIDTH = 20
SQUARE_HEIGHT = 20
# Margin between each square
SQUARE_MARGIN = 0
 
 
class Square(pygame.sprite.Sprite):
    def __init__(self, x, y, color):
        super().__init__()
 
        self.image = pygame.Surface([SQUARE_WIDTH, SQUARE_HEIGHT])
        self.image.fill(color)
 
        self.rect = self.image.get_rect()
        self.rect.x = x * (SQUARE_WIDTH + SQUARE_MARGIN) + SQUARE_MARGIN
        self.rect.y = y * (SQUARE_HEIGHT + SQUARE_MARGIN) + SQUARE_MARGIN
 

class VisibleRectangle(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height):
        super().__init__()
 
        rect_width = width * (SQUARE_WIDTH + SQUARE_MARGIN) + SQUARE_MARGIN
        rect_height = height * (SQUARE_HEIGHT + SQUARE_MARGIN) + SQUARE_MARGIN

        self.image = pygame.Surface([rect_width, rect_height])
        self.image.fill(GREY)
 
        self.rect = self.image.get_rect()
        self.rect.x = x * (SQUARE_WIDTH + SQUARE_MARGIN)
        self.rect.y = y * (SQUARE_HEIGHT + SQUARE_MARGIN)       


class Player():
    def __init__(self, x, y, color=WHITE):
        self.x = x
        self.y = y

        self.vx = 0
        self.vy = 0

        self.vxmax = 1
        self.vymax = 1

        self.grounded = False

    def update(self, x, y):
        self.x = x
        self.y = y

    def square(self):
        return Square(self.x, self.y, WHITE)


class Game():
    """
    Mario-like platforming game for an experiment on procedural content generation using generative adversarial networks and reinforcement learning. 
    """
    def __init__(self, width, height):
        
        self.clock = pygame.time.Clock()

        self.value_dict = {
            "air": 0,
            "player": 1,
            "ground": 2,
            "water": 3,
            "hazard": 4
        }

        self.world = np.zeros((width, height))
        self.world[:, -3:] = self.value_dict["ground"]
        self.world[13:18, -4] = self.value_dict["ground"]
        self.world[4:9, -4] = self.value_dict["ground"] # Does not work yet
        #self.world[18, -5] = self.value_dict["ground"]
        self.world[2, -4] = self.value_dict["ground"]
        self.world[19, -4] = self.value_dict["ground"]
        self.world[1, -5] = self.value_dict["ground"]

        self.width = width
        self.height = height

        self.screen = pygame.display.set_mode([width * (SQUARE_WIDTH + SQUARE_MARGIN) + SQUARE_MARGIN, 
                                               height * (SQUARE_HEIGHT + SQUARE_MARGIN) + SQUARE_MARGIN])

        self.sprite_group = pygame.sprite.Group()
        self.sprite_group.empty()

        self.player = Player(width // 2 + 9, height // 2, WHITE)
        self.player.vy = -1
        self.player.vx = -0.24

        self.collision_snap_distance = 0.0001

        self.g = 0.1

    def show(self):
        """
        Render the world. 
        """
        self.screen.fill(BLACK)

        self.sprite_group.empty()

        for x, y in itertools.product(range(self.width), range(self.height)):

            if self.world[x, y] == self.value_dict["ground"]:
                square = Square(x, y, GREY)
                self.sprite_group.add(square)

            if self.world[x, y] == self.value_dict["water"]:
                square = Square(x, y, BLUE)
                self.sprite_group.add(square)

            if self.world[x, y] == self.value_dict["hazard"]:
                square = Square(x, y, RED)
                self.sprite_group.add(square)

        self.sprite_group.add(self.player.square())

        self.sprite_group.draw(self.screen)

        pygame.display.flip()

        self.clock.tick(30)

    def tick(self):
        self.world = np.zeros((self.width, self.height))
        self.world[:, -3:] = self.value_dict["ground"]
        self.world[13:18, -4] = self.value_dict["ground"]
        self.world[13:18, -5] = self.value_dict["ground"]
        self.world[4:9, -4] = self.value_dict["ground"] # Does not work yet
        #self.world[18, -5] = self.value_dict["ground"]

        self.world[2, -4] = self.value_dict["ground"]
        self.world[19, -4] = self.value_dict["ground"]
        self.world[19, -5] = self.value_dict["ground"]

        #self.world[19, -5] = self.value_dict["ground"]
        self.world[1, -5] = self.value_dict["ground"]

        self.move_player()

    def move_player(self):
        """
        Update the player's velocity and move the player along the vector defined by its velocity (vx, vy). If 
        any block is encountered along the way, alter the trajectory accordingly.
        """

        # Change velocity depending on whether the player is on the ground or in the air
        if not self.player.grounded:
            self.player.vy += self.g
        else:
            if self.player.vx != 0:
                self.player.vx -= self.player.vx * 0.1

        # Clip the velocity
        self.player.vx = min(self.player.vxmax, self.player.vx)
        self.player.vy = min(self.player.vymax, self.player.vy)
        self.player.vx = max(-self.player.vxmax, self.player.vx)
        self.player.vy = max(-self.player.vymax, self.player.vy)

        # Compute candidate target point
        newx = self.player.x + self.player.vx
        newy = self.player.y + self.player.vy

        # Alter trajectory
        self.handle_movement_events(newx, newy)

    def detect_fall_event(self, newx):
        """
        Detects any fall off a block and returns the time of the fall.
        """

        if newx > self.player.x:
            on_block = (math.floor(self.player.x), math.floor(self.player.y) + 1)
            new_on_block = (math.floor(newx), math.floor(self.player.y) + 1)
        elif newx < self.player.x:
            on_block = (math.floor(self.player.x) + 1, math.floor(self.player.y) + 1)
            new_on_block = (math.floor(newx) + 1, math.floor(self.player.y) + 1)
        else:
            on_block = (math.floor(self.player.x), math.floor(self.player.y) + 1)
            if self.world[on_block] == self.value_dict["air"]:
                return (0, "fall")
            else:
                return (1, None)

        on_block = tuple(map(int, on_block))
        new_on_block = tuple(map(int, new_on_block))

        on_blocks = [(i, on_block[1]) for i in range(on_block[0], new_on_block[0] + (-1) ** (newx - self.player.x < 0), (-1) ** (newx - self.player.x < 0))]
        on_blocks = [idx for idx in on_blocks if (0 <= idx[0] < self.width) and (0 <= idx[1] < self.height)]

        air_blocks = [idx for idx in on_blocks if self.world[idx] == self.value_dict["air"] and idx != on_block]

        if air_blocks:
            first_idx = air_blocks[0]
            tcol = (first_idx[0] - self.player.x) / self.player.vx
            event_type = "fall"
        else:
            tcol = 1
            event_type = None

        return (tcol, event_type)

    def detect_collision_event(self, newx, newy):
        """
        Detects any collisions based on the old and new positions of the player and returns the time of collision
        and the direction the collision is in.
        """

        # Closest block to the middle of the player
        close_idx = (math.floor(newx + 1.), math.floor(newy + 1.))

        # 5 by 5 grid around the closest block
        close_idx = [(close_idx[0] + i - 2, close_idx[1] + j - 2) for i, j in itertools.product(range(5), range(5))]

        # Close blocks that are solid
        close_idx = [idx for idx in close_idx if (0 <= idx[0] < self.width) & (0 <= idx[1] < self.height)]
        close_idx = [idx for idx in close_idx if self.world[idx] == self.value_dict["ground"] or self.world[idx] == self.value_dict["hazard"]]

        # Collision if: right side is past left side, left side not past right side, and likewise for top and bottom
        collisions = {
            idx: (newx + 1 + self.collision_snap_distance > idx[0]) & (newx < idx[0] + 1 + self.collision_snap_distance) &
                 (newy + 1 + self.collision_snap_distance > idx[1]) & (newy < idx[1] + 1 + self.collision_snap_distance)
            for idx in close_idx
            }

        collisions = [idx for idx, collision in collisions.items() if collision]

        if self.player.x == newx:
            collisions = [idx for idx in collisions if idx[0] == newx]

        if self.player.y == newy:
            collisions = [idx for idx in collisions if idx[1] == newy]

        tcol = 1
        event_type = None
        if collisions:

            for idx in collisions:

                if (self.player.x + 1 > idx[0]) & (self.player.x < idx[0] + 1):
                    tcolx = 0.
                else:
                    if self.player.vx > 0:
                        tcolx = abs((idx[0] - self.player.x - 1) / self.player.vx)
                    elif self.player.vx < 0:
                        tcolx = abs((idx[0] + 1 - self.player.x) / self.player.vx)

                # Time to collision in y direction
                if (self.player.y + 1 > idx[1]) & (self.player.y < idx[1] + 1):
                    tcoly = 0.
                else:
                    if self.player.vy > 0:
                        tcoly = abs((idx[1] - self.player.y - 1) / self.player.vy)
                    elif self.player.vy < 0:
                        tcoly = abs((idx[1] + 1 - self.player.y) / self.player.vy)

                # Longest time determines the side that is hit
                if tcolx > tcoly:
                    if tcolx < tcol:
                        tcol = tcolx
                        event_type = 'x'
                elif tcolx < tcoly:
                    if tcoly < tcol:
                        tcol = tcoly
                        event_type = 'y'
                else:
                    tcol = tcolx
                    if 0 < abs(self.player.vx) < 3 * abs(self.player.vy):
                        event_type = 'x'
                    else:
                        event_type = 'y'

                        if abs(self.player.vx) > 0:
                            #self.player.y += self.collision_snap_distance * (-1) ** (self.player.vy > 0)
                            self.player.x += self.collision_snap_distance * (-1) ** (self.player.vx < 0)
                            self.player.grounded = False

        return (tcol, event_type)

    def handle_movement_events(self, newx, newy):
        """
        Handles any movement event between two timesteps. If multiple events happen in succession, they 
        are all handled successively.
        """

        if self.player.vx != 0 or self.player.vy != 0:
            collision_event = self.detect_collision_event(newx, newy)
        else:
            collision_event = (1, None)

        if self.player.grounded:
            fall_event = self.detect_fall_event(newx)
            event_t, event_type = min((collision_event, fall_event), key=lambda x: x[0])
        else:
            event_t, event_type = collision_event     

        t_total = 0.
        while event_type:
            t_total += event_t

            if event_type == 'x' or event_type == 'y':
                # Collision detected
                self.player.x += event_t * self.player.vx
                self.player.y += event_t * self.player.vy

                if abs(self.player.x % 1) < self.collision_snap_distance:
                    self.player.x = math.floor(self.player.x + .5)
                if abs(self.player.y % 1) < self.collision_snap_distance:
                    self.player.y = math.floor(self.player.y + .5)

                if event_type == 'x':
                    self.player.vx = 0
                elif event_type == 'y':
                    if self.player.vy >= 0 and event_t >= 0:
                        self.player.grounded = True

                    self.player.vy = 0

            else:
                # Fall detected
                self.player.x += event_t * self.player.vx
                self.player.grounded = False
                self.player.vy = .5 * (1 - t_total) * self.g

            # Compute updated target location
            newx = self.player.x + (1 - t_total) * self.player.vx
            newy = self.player.y + (1 - t_total) * self.player.vy

            if self.player.vx != 0 or self.player.vy != 0:
                collision_event = self.detect_collision_event(newx, newy)
            else:
                collision_event = (1, None)

            if self.player.grounded:
                fall_event = self.detect_fall_event(newx)
                event_t, event_type = min((collision_event, fall_event), key=lambda x: x[0])
            else:
                event_t, event_type = collision_event

        self.player.x = newx
        self.player.y = newy



# The idea is: 
# Repeat until the whole frame is done:
#   Find the first event, either fall or collision
#   Calculate position + velocity up to first event

if __name__ == "__main__": 
    
    pygame.init()
    pygame.display.set_caption('Snake')
    game = Game(21, 21)

    done = False

    while not done:
        game.show()

        game.tick()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
    
    pygame.quit()