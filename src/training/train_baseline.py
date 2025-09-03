# src/training/train_baseline.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

FEATURES_CSV = 'data/features/features.csv'
MODELS_DIR = 'models'

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

df = pd.read_csv(FEATURES_CSV)
X = df.drop('label', axis=1)
y = df['label']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# Train RandomForest
clf = RandomForestClassifier(
    n_estimators=200,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model, label encoder, scaler
joblib.dump(clf, os.path.join(MODELS_DIR, 'best_baseline.pkl'))
joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder.pkl'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))

print(f"Model, label encoder, and scaler saved in {MODELS_DIR}")
