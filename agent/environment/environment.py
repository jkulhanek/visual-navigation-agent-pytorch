import abc

class Environment:
    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def start(self):
        pass

    @abc.abstractproperty
    def actions(self):
        pass