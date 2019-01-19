# https://stackabuse.com/k-nearest-neighbors-algorithm-
# in-python-and-scikit-learn/

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Importing the Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-" \
      "databases/libras/movement_libras.data"

# Assign colum names to the dataset
names = ['curved swing', 'horizontal swing', 'vertical swing', 
         'anti-clockwise arc', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names)

# Preprocessing
X = np.array(dataset.iloc[:, :-1].values)
Y = np.array(dataset.iloc[:, 4].values)

# training and test splits
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

# feature scaling:
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# training und predictions
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)

# make predictions
y_pred = classifier.predict(X_test)

error = []

# Fehler zwischen 1 und 50
for i in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

# Fehler im Plot darstellen
plt.figure(figsize=(12, 6))
plt.plot(range(1, 50), error, color='red', linestyle='dashed', 
         marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
