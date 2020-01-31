import ai2thor.controller
from agent.environment.environment import Environment

class ThorEnvironment(Environment):
    def __init__(self, scene_name = 'FloorPlan28', unity_path = None, **kwargs):
        super(ThorEnvironment, self).__init__()
        self.controller = ai2thor.controller.Controller()
        self.controller.local_executable_path = unity_path
        self.is_started = False
        self.frame = None

    def _set_state(self, event):
        self.frame = event.frame

    def _initialize(self):
        self.controller.start()
        self.controller.reset('FloorPlan29')
        event = self.controller.step(dict(action='Initialize', gridSize=0.25))
        self._set_state(event)

    def reset(self, initial_state_id = None):
        if not self.is_started:
            self._initialize()

    def step(self, action = 'MoveAhead'):
        event = self.controller.step(dict(action=action))
        self._set_state(event)

    def render(self, mode):
        if mode == 'image':
            return self.frame
        else:
            assert False

    @property
    def actions(self):
        return ["MoveAhead", "RotateRight", "RotateLeft", "MoveBackward"]