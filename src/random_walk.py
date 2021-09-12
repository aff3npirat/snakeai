class RandomWalk:

    def __init__(self, n):
        self.n = n
        self.terminate_states = [0, n + 1]
        self.state = n // 2
        self.score = 0

    def play_step(self, action, render_game):
        self.state += 1 if action == 0 else -1
        if self.state in self.terminate_states:
            if self.state == self.terminate_states[1]:
                self.score += 1
                return [True, 1]
            return [True, 0]
        return [False, 0]

    def reset(self):
        self.state = self.n // 2
        self.score = 0

