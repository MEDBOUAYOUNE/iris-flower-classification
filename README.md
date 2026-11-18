# Iris Flower Classification

## The problem

The [Iris dataset](https://archive.ics.uci.edu/dataset/53/iris) contains 150 flowers from three species: **Iris-setosa**, **Iris-versicolor** and **Iris-virginica**. Each flower has four measurements (in cm):

- sepal length
- sepal width
- petal length
- petal width

The goal is to build a model that predicts the species of a flower from these four measurements. This is a multi-class classification problem.

## What the code does (`classify.py`)

1. **Load data**: reads `iris/iris.data` with pandas. The file has no header row, so column names are given manually.
2. **Explore data**: prints the first rows, the dataset info and the number of flowers per species.
3. **Visualize** (commented out by default): a boxplot of a feature and a scatter plot of petal length vs petal width, colored by species.
4. **Split data**: 80% for training, 20% for testing (`train_test_split`, `random_state=42` for reproducible results).
5. **Train model**: fits a `LogisticRegression` classifier on the training set.
6. **Predict**: predicts the species of the test flowers.
7. **Evaluate**: prints accuracy (share of correct predictions) and macro-averaged precision (how often a predicted species is correct, averaged over the three species).
8. **Plot predictions**: scatter of the test set (petal length vs petal width), colored by predicted species.

## Setup and run

```
pip install -r requirments.txt
python classify.py
```

The dataset must be at `iris/iris.data`.

## Libraries

pandas, numpy, matplotlib, scikit-learn (versions pinned in `requirments.txt`).
