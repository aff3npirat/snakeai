import pygame

from src.snake_game import SnakeGame, Direction, TILE_SIZE


class NaiveAgent:
    """An agent that follows a constant policy.

    The Agent will always choose the action which reduces the distance to the food, while avoiding death.
    """

    def get_state(self, game):
        pos_u = [game.head_position[0], game.head_position[1] - TILE_SIZE]
        pos_d = [game.head_position[0], game.head_position[1] + TILE_SIZE]
        pos_l = [game.head_position[0] - TILE_SIZE, game.head_position[1]]
        pos_r = [game.head_position[0] + TILE_SIZE, game.head_position[1]]

        state = [game.is_collision(pos_u),
                 game.is_collision(pos_d),
                 game.is_collision(pos_l),
                 game.is_collision(pos_r),

                 game.food_position,
                 game.head_position]
        return state

    def get_action_score(self, state, action):
        food = state[-2]
        head = state[-1]
        if not state[action]:
            if action == 0:
                d = abs(head[0]//TILE_SIZE - food[0]//TILE_SIZE) + abs(head[1]//TILE_SIZE - 1 - food[1]//TILE_SIZE)
            elif action == 1:
                d = abs(head[0]//TILE_SIZE - food[0]//TILE_SIZE) + abs(head[1]//TILE_SIZE + 1 - food[1]//TILE_SIZE)
            elif action == 2:
                d = abs(head[0]//TILE_SIZE - 1 - food[0]//TILE_SIZE) + abs(head[1]//TILE_SIZE - 1 - food[1]//TILE_SIZE)
            else:
                d = abs(head[0]//TILE_SIZE + 1 - food[0]//TILE_SIZE) + abs(head[1]//TILE_SIZE + 1 - food[1]//TILE_SIZE)
            q = 10 - d
        else:
            q = -100
        return q

    def get_action(self, state):
        q_max = self.get_action_score(state, 0)
        action = 0
        for i in range(1, 4):
            q = self.get_action_score(state, i)
            if q > q_max:
                q_max = q
                action = i
        return Direction(action)


if __name__ == '__main__':
    game = SnakeGame(20, 20)
    agent = NaiveAgent()
    done = False

    while not done:
        state = agent.get_state(game)
        done, reward, score = game.play_step(agent.get_action(state))

    print(f"Score: {score}")
    input()
    pygame.quit()
    quit()
