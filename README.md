# MNIST Classifier Project

This project implements two machine learning classifiers, K-Nearest Neighbors (KNN) and Naive Bayes, to classify handwritten digits from the MNIST dataset. The project is structured to handle data loading, preprocessing, and classification, with a focus on simplicity and clarity.

## Project Structure

```
MNIST-classifier/
├── src/
│   ├── classifiers/
│   │   ├── Classifier.py         # Base classifier interface
│   │   ├── KNNClassifier.py      # K-Nearest Neighbors implementation
│   │   └── NaiveBayesClassifier.py # Naive Bayes implementation
│   └── data/
│       ├── Dataset.py            # Dataset handling utilities
│       └── MNISTDataScaler.py    # Data preprocessing and scaling
├── data/
│   ├── train/
│   │   └── mnist_train.csv       # Training data
│   └── test/
│       └── mnist_test.csv        # Testing data
└── main.py                       # Main script to run classifiers
```

## Prerequisites

- Python 3.6+
- Required libraries: `numpy`, `pandas`, `scikit-learn`

Install dependencies using:
```bash
pip install numpy pandas scikit-learn
```

## Usage

1. **Prepare the Data**:
   - Place the MNIST dataset files (`mnist_train.csv` and `mnist_test.csv`) in the `data/train/` and `data/test/` directories, respectively.
   - The dataset should have 784 feature columns for pixel values (unlabeled data) or 785 columns including the label (labeled data).

2. **Run the Classifiers**:
   - The `main.py` script runs both the Naive Bayes and KNN classifiers on the provided dataset.
   - Execute the script from the project root directory:
     ```bash
     python main.py
     ```
   - By default, the KNN classifier uses `k=5`. You can modify the `k` value in the `main.py` script if needed.

3. **Output**:
   - The script will output:
     - The classifier name.
     - Indices where predictions differ from expected labels, showing expected and predicted values.
     - The number of correct predictions and total samples.
     - The accuracy percentage.
     - The computation time in minutes.

## Classifiers

- **Naive Bayes** (`NaiveBayesClassifier.py`):
  - Implements a Naive Bayes classifier with binarization (threshold at 128) and Laplace smoothing to handle zero probabilities.
  - Assumes 10 classes (digits 0-9) and 784 features (28x28 pixel images).

- **K-Nearest Neighbors** (`KNNClassifier.py`):
  - Implements a KNN classifier with Euclidean distance and weighted voting based on inverse distances.
  - Uses `MNISTDataScaler` to normalize pixel values to the range [0, 1].
  - Adjusts `k` to an odd number if an even value is provided to avoid ties.

## Data Handling

- **Dataset.py**:
  - Provides a `Dataset` class to load and manage labeled or unlabeled MNIST data from CSV files.
  - Automatically detects whether the data is labeled (785 columns) or unlabeled (784 columns).

- **MNISTDataScaler.py**:
  - Scales pixel values from [0, 255] to [0, 1] for KNN classification.
  - Supports forward and inverse transformations.

## Classifier Performance Comparison

The following table compares the performance of the Naive Bayes and KNN classifiers on the MNIST test dataset (10,000 samples). Performance metrics are based on typical runs with the provided implementation and may vary slightly depending on system specifications.

| Classifier   | Accuracy (%) | Runtime (minutes) | Notes                              |
|--------------|--------------|-------------------|------------------------------------|
| Naive Bayes  | ~83-85       | ~0.5-1.0          | Fast, uses binarized features      |
| KNN (k=5)    | ~96-97       | ~5-10             | Slower, uses normalized features   |

**Notes**:
- Accuracy values are approximate and depend on the specific MNIST dataset split.
- Runtime is measured on a standard CPU; KNN is computationally intensive due to distance calculations.
- Naive Bayes is faster but less accurate due to its simplifying assumptions.
- KNN achieves higher accuracy but requires more computation time, especially for larger `k` values.

## Notes

- The project assumes the MNIST dataset is in CSV format with pixel values as unsigned 8-bit integers.
- The `Classifier.py` file defines an abstract base class for classifiers, ensuring a consistent interface for `fit`, `predict`, and `evaluate` methods.
- Logging is enabled in `KNNClassifier.py` to track prediction progress.