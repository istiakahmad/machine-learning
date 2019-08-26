import pandas as pd
# dataset (csv file) path
path = "crx.csv"


# reading the csv
data = pd.read_csv(path)


def get_features_and_label(data_from_csv):
    data_from_csv.dropna(inplace=True)
    features = data_from_csv.iloc[:, :-1]  # Extract features from the Dataset X1-X23, except the first ID column
    label = data_from_csv.iloc[:, -1]  # Extract the labels from the Dataset.
    return features, label


features, label = get_features_and_label(data)

print(features)
print(label)
