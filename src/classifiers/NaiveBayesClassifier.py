import numpy as np

from classifiers.Classifier import Classifier
import numpy as np

from classifiers.Classifier import Classifier


class NaiveBayesClassifier(Classifier):
    def __init__(self):
        self.bin_threshold = 128
        self.num_classes = 10
        self.num_features = 784
        self._features = None
        self._labels = None
        self._prior_probabilities = None
        self._likelihood_probabilities = None

    def binarize(self, features):
        return (features >= self.bin_threshold).astype(np.uint8)*255

    def calculate_prior_probabilities(self):
        self._prior_probabilities = np.zeros(self.num_classes)
        #caluclate the number of each classes
        for label in self._labels:
            self._prior_probabilities[label] += 1
        #get the probability by dividing each class count by the total number
        self._prior_probabilities /= len(self._labels)

    def calculate_likelihood_probabilities(self):
        self._likelihood_probabilities = np.zeros((self.num_classes, self.num_features))
        for label, feature in zip(self._labels, self._features):
            for feature_index in range(self.num_features):
                # Check if the feature value is 255 and update the count
                if feature[feature_index] == 255:
                    self._likelihood_probabilities[label, feature_index] += 1
        #get the probability by dividing by the number counts by the of samples of each class
        #apply laplace smoothing to avoid zero probability
        for label in range(self.num_classes):
            class_count = np.sum(self._labels == label)
            if class_count!=0:
                self._likelihood_probabilities[label] = (self._likelihood_probabilities[label]+1)/ (class_count+2)

    def fit(self, dataset):
        if dataset._isLabeled == False:
            raise ValueError("Dataset must be labeled")
        self._labels = dataset.getLabels()
        self._features = self.binarize(dataset.getFeatures())
        self.calculate_prior_probabilities()
        self.calculate_likelihood_probabilities()

    def predict(self, dataset):
        if self._features is None or self._labels is None:
            raise ValueError("Classifier must be fitted first")
        features_to_predict = self.binarize(dataset.getFeatures())
        predicted_labels = np.zeros(len(features_to_predict), dtype=np.uint8)

        log_prior = np.log(self._prior_probabilities + 1e-9)
        log_likelihood = np.log(self._likelihood_probabilities + 1e-9)
        log_likelihood_complement = np.log(1 - self._likelihood_probabilities )

        for i, feature_to_predict in enumerate(features_to_predict):
            log_probs = np.copy(log_prior)
            for label in range(self.num_classes):
                log_probs[label] += np.sum(
                    (feature_to_predict == 255) * log_likelihood[label] +
                    (feature_to_predict == 0) * log_likelihood_complement[label]
                )
            predicted_labels[i] = np.argmax(log_probs)
        return predicted_labels


    def evaluate(self, dataset):
        if self._features is None or self._labels is None:
            raise ValueError("Classifier must be fitted first")
        predicted_labels = self.predict(dataset)
        correct = np.sum(predicted_labels == dataset.getLabels())
        accuracy = (correct / len(predicted_labels))*100
        return accuracy


