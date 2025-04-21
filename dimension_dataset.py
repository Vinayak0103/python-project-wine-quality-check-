# visualize the wine quality data
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
dataset = pd.read_csv(url, sep=';')

# Box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(4, 3), sharex=False, sharey=False, figsize=(12, 10))
plt.tight_layout()
plt.show()

# Histograms
dataset.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

# Scatter plot matrix
scatter_matrix(dataset, figsize=(14, 12))
plt.show()
