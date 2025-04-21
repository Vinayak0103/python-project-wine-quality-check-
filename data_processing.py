import pandas as pd

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
dataset = pd.read_csv(url, sep=';')

# 1. Check for missing values
print("Missing values per column:\n", dataset.isnull().sum())

# 2. Drop any rows with missing values (if any)
dataset = dataset.dropna()

# 3. Check for duplicates
print("Number of duplicate rows:", dataset.duplicated().sum())

# 4. Remove duplicates if any
dataset = dataset.drop_duplicates()

# 5. Check summary statistics to see if scaling or normalization is needed
print("\nSummary Statistics:\n", dataset.describe())

# 6. Optional: Scaling numerical features (standardization or normalization)
# You can use StandardScaler or MinMaxScaler for scaling. Example using MinMaxScaler:

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Scale all features except 'quality' (target variable)
features = dataset.drop("quality", axis=1)
scaled_features = scaler.fit_transform(features)

# Create a new DataFrame with scaled features
scaled_dataset = pd.DataFrame(scaled_features, columns=features.columns)
scaled_dataset['quality'] = dataset['quality']

# 7. Check the cleaned dataset
print("\nCleaned Dataset (first 5 rows):\n", scaled_dataset.head())
