import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("creditcard.csv")

# We'll use 'Amount' and 'Time' plus the PCA features V1-V28
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Train a simple model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate it (optional)
print(classification_report(y_test, model.predict(X_test)))

# Save the model
joblib.dump(model, 'fraud_model.pkl')

