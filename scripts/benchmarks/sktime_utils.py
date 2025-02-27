from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datasets import load_arrow_head

# step 1-- prepare a dataset (multivariate for demonstration)
X_train, y_train = load_arrow_head(split="train")
X_new, _ = load_arrow_head(split="test")

# step 2-- define the TimeSeriesForestClassifier
tsf = TimeSeriesForestClassifier()

# step 3-- train the classifier on the data
tsf.fit(X_train, y_train)

# step 4-- predict labels on the new data
y_pred = tsf.predict(X_new[:5])

print(type(y_pred),y_pred.shape,y_pred)

