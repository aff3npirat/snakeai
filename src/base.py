class AgentBase:
    """Captures environment in states."""

    def __init__(self, model):
        self.model = model

    def get_state(self, game):
        raise NotImplementedError


# class TrainerBase:
#     """Trains a specific model."""
#
#     def __init__(self, model):
#         self.model = model
#
#     def train_step(self, episode):
#         raise NotImplementedError


class ModelBase:
    """Selects an action based on the representation (state) of environment.

    All keyword arguments passed to __init__ can be accessed like normal instance attributes.
    """

    def __init__(self, **kwargs):
        # contains all parameters which are saved to .yml file.
        self.params = kwargs

    def __getattr__(self, name):
        if name in self.__dict__['params']:
            return self.__dict__['params'][name]
        else:
            raise AttributeError

    def __setattr__(self, name, value):
        if not name == 'params' and name in self.params:
            self.params[name] = value
        else:
            self.__dict__[name] = value

    def __setstate__(self, state):
        self.__dict__.update(state)

    def get_action(self, world_state):
        raise NotImplementedError
