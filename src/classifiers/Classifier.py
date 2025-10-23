import abc

class Classifier(abc.ABC):
    @abc.abstractmethod
    def predict(self, dataset):
        pass
    @abc.abstractmethod
    def fit(self, dataset):
        pass

    @abc.abstractmethod
    def evaluate(self, dataset):
        pass


    # @abc.abstractmethod
    # def save(self, path):
    #     pass
    # @abc.abstractmethod
    # def load(self, path):
    #     pass