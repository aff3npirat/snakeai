from snakeai.helper import write_to_file


class AgentBase:
    """Captures environment in states."""

    def __init__(self, model, trainer):
        self.model = model
        self.trainer = trainer

    def save(self, root_dir, agent_name):
        write_to_file(self, root_dir / f"{agent_name}.pkl")
        params = self.model.params
        params.update(self.trainer.params)
        write_to_file(params, root_dir / f"{agent_name}.yml")


class ModelBase:
    """Selects an action based on the representation (state) of environment.

    All keyword arguments passed to __init__ can be accessed like normal instance attributes.
    """

    def __init__(self, Q, **kwargs):
        # contains all parameters which are saved to .yml file.
        self.params = kwargs
        self.Q = Q

    def __getattr__(self, name):
        if name in self.__dict__['params']:
            return self.__dict__['params'][name]
        else:
            raise AttributeError

    def __setattr__(self, name, value):
        if name != 'params' and name in self.params:
            self.params[name] = value
        else:
            self.__dict__[name] = value

    def __setstate__(self, state):
        self.__dict__.update(state)

    def get_action(self, world_state):
        raise NotImplementedError


class TrainerBase:

    def __init__(self, Q, **kwargs):
        self.params = kwargs
        self.Q = Q

    def __getattr__(self, name):
        if name in self.__dict__['params']:
            return self.__dict__['params'][name]
        else:
            raise AttributeError

    def __setattr__(self, name, value):
        if name != 'params' and name in self.params:
            self.params[name] = value
        else:
            self.__dict__[name] = value

    def __setstate__(self, state):
        self.__dict__.update(state)
