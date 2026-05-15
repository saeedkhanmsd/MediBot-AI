import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

os.makedirs('models', exist_ok=True)

print("="*50)
print("🤖 TRAINING MODEL WITH KAGGLE DATASET")
print("="*50)

# Load
df = pd.read_csv('data/large_dataset.csv')
print(f"✅ Loaded: {len(df)} rows")

# Combine symptoms
symptom_cols = [col for col in df.columns if 'Symptom' in col]
symptoms_list = []

for idx, row in df.iterrows():
    syms = []
    for col in symptom_cols:
        val = row[col]
        if pd.notna(val) and str(val).strip():
            syms.append(str(val).strip().replace('_', ' '))
    symptoms_list.append(', '.join(syms))

final_df = pd.DataFrame({
    'disease': df['Disease'],
    'symptoms': symptoms_list
})

print(f"✅ Processed: {len(final_df)} rows, {final_df['disease'].nunique()} diseases")

# Train
vectorizer = TfidfVectorizer(max_features=300)
X = vectorizer.fit_transform(final_df['symptoms'])
y = final_df['disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")

# Save
joblib.dump(model, 'models/disease_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
print("✅ Model saved!")

print("\n🧪 Quick Test:")
tests = ["itching skin rash", "fever chills headache", "runny nose sneezing"]
for t in tests:
    vec = vectorizer.transform([t])
    pred = model.predict(vec)[0]
    print(f"   '{t}' → {pred}")