from snakeai.helper import write_to_file, dict_to_string


class AgentBase:

    def __init__(self, Q, eps_greedy, **kwargs):
        self.Q = Q
        self.eps_greedy = eps_greedy
        self.n_games = 0
        self.params = kwargs

    def get_action(self, state):
        raise NotImplementedError

    def save(self, root, agent_name):
        write_to_file(self, root / f"{agent_name}.pkl")
        write_to_file(dict_to_string(self.params), root / f"{agent_name}.yml", text=True)

    def __getattr__(self, name):
        if name in self.__dict__['params']:
            return self.__dict__['params'][name]
        else:
            raise AttributeError

    def __setattr__(self, name, value):
        if name != 'params' and name in self.__dict__.get('params', []):
            self.params[name] = value
        else:
            self.__dict__[name] = value

    def __setstate__(self, state):
        self.__dict__ = state
