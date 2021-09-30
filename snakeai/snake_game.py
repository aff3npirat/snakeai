import pygame
import random

# movement directions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# colors
BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)

# game properties
TILE_SIZE = 10
SNAKE_SIZE = 8
SPEED = 30


class SnakeGame:
    def __init__(self, x_tiles, y_tiles, render):
        self.x_tiles = x_tiles
        self.y_tiles = y_tiles
        self.fps = pygame.time.Clock()
        self.direction = RIGHT
        self.score = 0
        self.n_steps = 1
        self.head = []
        self.body = []
        self.food = []
        self.reset()
        if render:
            pygame.init()
            pygame.display.set_caption("Snake Game")
            self.game_window = pygame.display.set_mode((x_tiles * TILE_SIZE, y_tiles * TILE_SIZE))
            self.update_ui()
        else:
            self.game_window = None

    def reset(self):
        self.direction = RIGHT
        self.score = 0
        self.n_steps = 1
        self.head = [(self.x_tiles // 2) * TILE_SIZE, (self.y_tiles // 2) * TILE_SIZE]
        self.body = [[self.head[0] - i*TILE_SIZE, self.head[1]] for i in range(1, 4)]
        self.food = [random.randrange(0, self.x_tiles) * TILE_SIZE,
                     random.randrange(0, self.y_tiles) * TILE_SIZE]

    def play_step(self, action):
        self.body.insert(0, list(self.head))  # pass head by value, not reference
        self.n_steps += 1

        if action == UP and not self.direction == DOWN:
            self.direction = action
        elif action == DOWN and not self.direction == UP:
            self.direction = action
        elif action == LEFT and not self.direction == RIGHT:
            self.direction = action
        elif action == RIGHT and not self.direction == LEFT:
            self.direction = action

        if self.direction == UP:
            self.head[1] -= TILE_SIZE
        elif self.direction == DOWN:
            self.head[1] += TILE_SIZE
        elif self.direction == LEFT:
            self.head[0] -= TILE_SIZE
        elif self.direction == RIGHT:
            self.head[0] += TILE_SIZE

        reward = 0
        done = False
        if self.head == self.food:
            self.score += 1
            self.n_steps = 1
            reward += 10
            self.food = [random.randrange(0, self.x_tiles) * TILE_SIZE,
                         random.randrange(0, self.y_tiles) * TILE_SIZE]
        else:
            self.body.pop()

        if len(self.body)+1 == self.x_tiles * self.y_tiles:
            # victory
            reward += 10
            done = True
        if self.is_collision(self.head) or self.n_steps == self.x_tiles * self.y_tiles:
            # lose
            reward -= 10
            done = True

        if self.game_window is not None:
            self.update_ui()
            self.fps.tick(SPEED)

        return [done, reward]

    def is_collision(self, point):
        return self.out_of_bounds(point) or point in self.body

    def out_of_bounds(self, point):
        w = self.x_tiles * TILE_SIZE
        h = self.y_tiles * TILE_SIZE
        if point[0] < 0 or point[0] > w - TILE_SIZE:
            return True
        if point[1] < 0 or point[1] > h - TILE_SIZE:
            return True
        return False

    def update_ui(self):
        self.game_window.fill(BLACK)

        offset = (TILE_SIZE-SNAKE_SIZE) / 2
        # draw body
        for pos in self.body:
            pygame.draw.rect(
                self.game_window,
                GREEN,
                pygame.Rect(pos[0] + offset, pos[1] + offset, SNAKE_SIZE, SNAKE_SIZE)
            )
        # draw food
        pygame.draw.rect(
            self.game_window,
            WHITE,
            pygame.Rect(self.food[0], self.food[1], TILE_SIZE, TILE_SIZE)
        )
        # draw head
        left = self.head[0] + offset
        top = self.head[1] + offset
        pygame.draw.rect(
            self.game_window,
            RED,
            pygame.Rect(left, top, SNAKE_SIZE, SNAKE_SIZE)
        )
        pygame.display.update()
