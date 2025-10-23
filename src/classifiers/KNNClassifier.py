import numpy as np
import logging

from data.MNISTDataScaler import MNISTDataScaler

logging.basicConfig(level=logging.INFO)

from classifiers.Classifier import Classifier


class KNNClassifier(Classifier):
    def __init__(self, k):
        if k < 1:
            raise ValueError("K must be greater than 0")
        self._k = k
        self._features=None
        self._labels=None
        self._scaler = MNISTDataScaler()


    def fit(self, dataset):
        if dataset._isLabeled == False:
            raise ValueError("Dataset must be labeled")
        self._labels = dataset.getLabels()
        self._features = self._scaler.fit_transform(dataset.getFeatures())

    def predict(self, dataset, k=None):
        if self._features is None or self._labels is None:
            raise ValueError("Classifier must be fitted first")
        featuresToPredict = self._scaler.transform(dataset.getFeatures())
        predictedLabels = np.zeros(len(featuresToPredict), dtype=np.uint8)
        if k is None and self._k is None:
            raise ValueError("K must be specified")
        k=k or self._k
        if k % 2==0:
            k+=1

        for n, featureToPredict in enumerate(featuresToPredict, start=1):
            #distances=np.linalg.norm(self._features-featureToPredict,ord=2 ,axis=1)
            squaredDistances = np.sum((self._features - featureToPredict) ** 2, axis=1)
            kNearestIndices = np.argsort(squaredDistances)[:k]
            found_exact_match = False
            for i in kNearestIndices:
                if squaredDistances[i]==0:
                    predictedLabels[n-1] = self._labels[i]
                    found_exact_match = True
                    break
            if found_exact_match:
                continue

            #print(distances[kNearestIndices])
            kNearestLabels = self._labels[kNearestIndices]
            kNearestDistances = squaredDistances[kNearestIndices]

            #print(kNearestLabels)
            ############
            #predictedLabel = np.uint8(np.argmax(np.bincount(kNearestLabels)))
            ##################
            weights={}
            for label, distance in zip(kNearestLabels, kNearestDistances):
                weight = 1 / (distance)
                if label in weights:
                    weights[label] += weight
                else:
                    weights[label] = weight
            predictedLabel = max(weights, key=weights.get)



            predictedLabels[n-1] = predictedLabel
            progress_threshold = max(1, len(featuresToPredict) // 100)
            if n % progress_threshold == 0 or n == len(featuresToPredict):
                percent_complete = (n / len(featuresToPredict)) * 100
                logging.info(f"Processed {percent_complete:.0f}%")

        return np.array(predictedLabels, dtype=np.uint8)


    def evaluate(self, dataset):
        if self._features is None or self._labels is None:
            raise ValueError("Classifier must be fitted first")
        predictedLabels = self.predict(dataset)
        correct = np.sum(predictedLabels == dataset.getLabels())
        accuracy = (correct / len(predictedLabels))*100
        return accuracy













