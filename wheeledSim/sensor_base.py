import abc

class Sensor(abc.ABC):
    """
    Sensor base class
    """
    @abc.abstractmethod
    def measure(self):
        pass

    @abc.abstractmethod
    def to_rosmsg(self):
        pass
