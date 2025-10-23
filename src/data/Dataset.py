import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler


class Dataset:
    def __init__(self, features, labels=None):
        self._features = features        #nx28
        self._labels = labels            #n
        self._isLabeled = labels is not None
        if self._isLabeled and len(features) != len(labels):
            raise ValueError(f"Mismatch: {len(features)} features and {len(labels)} labels")

    @staticmethod
    def loadLabeled(filename):
        data = pd.read_csv(filename, dtype=np.uint8).values
        labels = data[:, 0]
        features = data[:, 1:]
        if len(features) != len(labels):
            raise ValueError(f"Mismatch: {len(features)} features and {len(labels)} labels")
        return features, labels

    @staticmethod
    def loadUnlabeled(filename):
        data = pd.read_csv(filename).values
        features = data
        return features, None

    @classmethod
    def from_file(cls,filename):
        data = pd.read_csv(filename)
        if len(data.columns) == 785:  #  785 columns = labeled data
            features, labels = cls.loadLabeled(filename)
        elif len(data.columns) == 784:  # 784 columns = unlabeled data
            features, labels = cls.loadUnlabeled(filename)
        else:
            raise ValueError(f"Unexpected number of columns: {len(data.columns)}")
        return cls(features, labels)



    def __str__(self):
        return (
            f"Labels shape: {self._labels.shape if self._labels is not None else 'None'}\n"
            f"Features shape: {self._features.shape}"
        )
    def getLabels(self):
        if self._isLabeled:
            return self._labels
        else:
            return np.zeros(len(self._features))

    def getFeatures(self):
        return self._features

    def isLabeled(self):
        return self._isLabeled




