import click
import pygame

from src.snake_game import SnakeGame, Direction


@click.command()
@click.option("-w",
              type=float)
@click.option("-h",
              type=float)
def main(w, h):
    game = SnakeGame(w, h, "Snake")
    done = False

    while not done:
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

        game.play_step(Direction(action))
    input()


if __name__ == '__main__':
    main()