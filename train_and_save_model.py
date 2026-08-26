import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# 1. Load preprocessed data (from Week 1)
df = pd.read_csv('titanic_small_preprocessed.csv')

X = df.drop(columns=['survived'])
y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 2. Train final model using the best hyperparameters found in Week 3 (GridSearchCV)
model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, preds):.3f}")
print(f"Test F1 Score: {f1_score(y_test, preds):.3f}")
print(f"Feature order expected by model: {list(X.columns)}")

# 3. Serialize model with joblib
joblib.dump(model, 'model.joblib')
print("Saved: model.joblib")

# Save the exact feature column order/names — the API needs this to build
# a correctly-ordered input row from JSON.
joblib.dump(list(X.columns), 'feature_columns.joblib')
print("Saved: feature_columns.joblib")
