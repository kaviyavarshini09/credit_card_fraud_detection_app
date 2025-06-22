import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Load your fraud detection dataset
data = pd.read_csv("creditcard.csv")

# Features and labels
X = data.drop("Class", axis=1)  # 'Class' is the target: 0 (non-fraud), 1 (fraud)
y = data["Class"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Save model
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save scaler (you need it during prediction)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
