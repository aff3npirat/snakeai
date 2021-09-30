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
    def __init__(self, x_tiles, y_tiles):
        self.x_tiles = x_tiles
        self.y_tiles = y_tiles
        self.game_window = pygame.display.set_mode((self.x_tiles * TILE_SIZE, self.y_tiles * TILE_SIZE))
        self.fps = pygame.time.Clock()
        pygame.init()
        pygame.display.set_caption("Snake Game")
        self.direction = RIGHT
        self.head_position = []
        self.body_position = []
        self.food_position = []
        self.score = 0
        self.n_steps = 0
        self.reset()

    # TODO: on each reset, snake should move in a random direction
    def reset(self):
        self.direction = RIGHT
        self.score = 0
        self.n_steps = 0
        self.head_position = [(self.x_tiles // 2) * TILE_SIZE, (self.y_tiles // 2) * TILE_SIZE]
        self.body_position = [[(self.x_tiles // 2) * TILE_SIZE - TILE_SIZE, (self.y_tiles // 2) * TILE_SIZE],
                              [(self.x_tiles // 2) * TILE_SIZE - 2 * TILE_SIZE, (self.y_tiles // 2) * TILE_SIZE],
                              [(self.x_tiles // 2) * TILE_SIZE - 3 * TILE_SIZE, (self.y_tiles // 2) * TILE_SIZE]]
        self.food_position = [random.randrange(1, self.x_tiles) * TILE_SIZE, random.randrange(1, self.y_tiles) * TILE_SIZE]

    def play_step(self, action, render=False):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()

        self.n_steps += 1

        self.body_position.insert(0, list(self.head_position))

        if action == UP and not self.direction == DOWN:
            self.direction = action
        elif action == DOWN and not self.direction == UP:
            self.direction = action
        elif action == LEFT and not self.direction == RIGHT:
            self.direction = action
        elif action == RIGHT and not self.direction == LEFT:
            self.direction = action

        if self.direction == UP:
            self.head_position[1] -= TILE_SIZE
        elif self.direction == DOWN:
            self.head_position[1] += TILE_SIZE
        elif self.direction == LEFT:
            self.head_position[0] -= TILE_SIZE
        elif self.direction == RIGHT:
            self.head_position[0] += TILE_SIZE

        reward = 0
        if self.head_position == self.food_position:
            self.score += 1
            self.n_steps = 0
            reward += 10
            self.food_position = [random.randrange(1, self.x_tiles) * TILE_SIZE, random.randrange(1, self.y_tiles) * TILE_SIZE]
        else:
            self.body_position.pop()

        if len(self.body_position)+1 == self.x_tiles * self.y_tiles:
            reward += 15
            return [True, reward]
        if self.is_collision(self.head_position) or self.n_steps == self.x_tiles * self.y_tiles:
            reward -= 10
            return [True, reward]

        if render:
            self.update_ui()
            self.fps.tick(SPEED)
        return [False, reward]

    def is_collision(self, point):
        return self.out_of_bounds(point) or self.is_body_position(point)

    def is_body_position(self, point):
        return point in self.body_position

    def out_of_bounds(self, point):
        w = self.x_tiles * TILE_SIZE
        h = self.y_tiles * TILE_SIZE
        if point[0] < 0 or point[1] > w - TILE_SIZE:
            return True
        if point[0] < 0 or point[1] > h - TILE_SIZE:
            return True

    def update_ui(self):
        self.game_window.fill(BLACK)
        pygame.draw.rect(self.game_window, RED, pygame.Rect(*self.head_position, SNAKE_SIZE, SNAKE_SIZE))
        for pos in self.body_position:
            pygame.draw.rect(self.game_window, GREEN, pygame.Rect(*pos, SNAKE_SIZE, SNAKE_SIZE))
        pygame.draw.rect(self.game_window, WHITE, pygame.Rect(self.food_position[0], self.food_position[1], TILE_SIZE, TILE_SIZE))
        pygame.display.update()
