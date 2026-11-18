import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load Data_Set (iris.data has no header row)

columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
data_set = pd.read_csv("./iris/iris.data", header=None, names=columns)

# Discover Data_Set

print(data_set.head())
print(data_set.info())
print(data_set["target"].value_counts())

# Visualize the features of the Data_set

# feature_1 = data_set["sepal_length"]
# plt.boxplot(feature_1)
# plt.xlabel("sepal_length")
# plt.ylabel("cm")
# plt.show()

# Visualize target scatter (one color per species)
colors = {"Iris-setosa": "red", "Iris-versicolor": "green", "Iris-virginica": "blue"}
for species, group in data_set.groupby("target"):
    plt.scatter(
        group["petal_length"],
        group["petal_width"],
        color=colors[species],
        label=species,
    )
plt.xlabel("petal_length (cm)")
plt.ylabel("petal_width (cm)")
plt.legend()
plt.show()

# Split Data_Set Train (80%), Test(20%)
x = data_set.loc[:, 'sepal_length':'petal_width']
y = data_set['target']
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

