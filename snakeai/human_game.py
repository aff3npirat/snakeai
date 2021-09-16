import click
import pygame

from snakeai.snake_game import SnakeGame, Direction


@click.command()
@click.option("-w",
              type=int)
@click.option("-h",
              type=int)
def main(w, h):
    game = SnakeGame(w, h, "Snake")
    done = False

    while not done:
        action = game.direction.value
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_RIGHT:
                    action = 3

        done, _ = game.play_step(Direction(action), True)
    input()


if __name__ == '__main__':
    main()