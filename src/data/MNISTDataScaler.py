import numpy as np
class MNISTDataScaler:
    def __init__(self):
        self._min=0
        self._max=255
        self._rangeMin=0
        self._rangeMax=1

    def fit_transform(self, dataset):
        if not isinstance(dataset, np.ndarray):
            raise ValueError("Input data must be a NumPy array")
        return (dataset-self._min)/(self._max-self._min)*(self._rangeMax-self._rangeMin)+self._rangeMin

    def transform(self, dataset):
        return self.fit_transform(dataset)

    def inverse_transform(self, dataset):
        return dataset*(self._rangeMax-self._rangeMin)+self._rangeMin
