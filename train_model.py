import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pickle

# Load sample dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Save model
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)
