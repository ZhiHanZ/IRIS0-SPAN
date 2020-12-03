from abc import ABC, abstractmethod


class DataSetInterface(ABC):
    @abstractmethod
    def load_data(self):
        pass
    @abstractmethod
    def load_testset(self):
        pass
    @abstractmethod
    def load_trainset(self):
        pass
    @abstractmethod
    def get_name(self) -> str :
        pass