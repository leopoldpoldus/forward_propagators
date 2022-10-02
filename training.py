import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import pickle

data = pd.read_csv('data.csv')
X = data[[str(i) for i in range(69)]]
y = data['cat']

# One-hot encode the labels
o = OneHotEncoder()
y = o.fit_transform(y.values.reshape(-1, 1)).toarray()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = RandomForestClassifier(max_depth=20, n_estimators=500, random_state=0)
clf.fit(X_train, y_train)

print(accuracy_score(y_test, clf.predict(X_test)))

# Save the model
with open('random_forest_classifier_v2.pkl', 'wb') as fid:
    pickle.dump(clf, fid)
