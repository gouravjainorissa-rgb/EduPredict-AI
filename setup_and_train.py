"""
Full setup: trains both classifier (pass/fail) and regressor (grade prediction)
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import joblib
import shap

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/student_data.csv")
X = df.drop(["pass", "final_score"], axis=1)
y_class = df["pass"]
y_reg   = df["final_score"]

X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

# ── Classifier ──────────────────────────────
print("Training classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, yc_train)
acc = accuracy_score(yc_test, clf.predict(X_test))
print(f"  Accuracy: {acc*100:.2f}%")
joblib.dump(clf, "models/model.pkl")

# SHAP explainer
explainer = shap.Explainer(clf, X_train)
joblib.dump(explainer, "models/explainer.pkl")
print("  Classifier + explainer saved.")

# ── Regressor ───────────────────────────────
print("Training grade regressor...")
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, yr_train)
mae = mean_absolute_error(yr_test, reg.predict(X_test))
r2  = r2_score(yr_test, reg.predict(X_test))
print(f"  MAE: {mae:.4f}  |  R²: {r2:.4f}")
joblib.dump(reg, "models/grade_model.pkl")
print("  Regressor saved.")

print("\n✅ All models trained and saved!")
