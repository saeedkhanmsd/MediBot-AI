import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

os.makedirs('models', exist_ok=True)

# Simple, accurate symptom-disease mapping
data = {
    'disease': [
        # Fever and related
        'Fever', 'Fever', 'Fever', 'Fever',
        'Malaria', 'Malaria', 'Malaria',
        'Common Cold', 'Common Cold', 'Common Cold', 'Common Cold',
        'COVID-19', 'COVID-19', 'COVID-19',
        
        # Pain related
        'Migraine', 'Migraine', 'Migraine',
        'Arthritis', 'Arthritis', 'Arthritis',
        'Headache', 'Headache',
        
        # Chronic
        'Diabetes', 'Diabetes', 'Diabetes',
        'Hypertension', 'Hypertension',
        
        # Stomach
        'Food Poisoning', 'Food Poisoning', 'Food Poisoning',
        'Acid Reflux', 'Acid Reflux',
        
        # Skin
        'Allergy', 'Allergy', 'Allergy',
        'Skin Rash', 'Skin Rash',
        
        # Respiratory
        'Asthma', 'Asthma',
        'Pneumonia', 'Pneumonia',
        
        # Urinary
        'UTI', 'UTI'
    ],
    'symptoms': [
        'high temperature body ache weakness',
        'fever chills sweating headache',
        'high fever headache muscle pain',
        'temperature fatigue body pain',
        'high fever chills sweating headache',
        'fever nausea vomiting muscle pain',
        'intermittent fever chills sweating',
        'runny nose sneezing sore throat',
        'cough nasal congestion mild fever',
        'sneezing runny nose watery eyes',
        'stuffy nose cough throat irritation',
        'fever dry cough loss of taste',
        'cough fever difficulty breathing',
        'dry cough fatigue sore throat',
        'severe headache nausea light sensitivity',
        'throbbing headache vomiting blurred vision',
        'headache nausea dizziness',
        'joint pain swelling stiffness',
        'joint inflammation pain reduced mobility',
        'knee pain stiffness morning',
        'headache pressure temples',
        'headache back head neck',
        'frequent urination excessive thirst fatigue',
        'blurred vision increased hunger weight loss',
        'fatigue slow healing numbness',
        'headache chest pain dizziness',
        'high blood pressure fatigue irregular heartbeat',
        'vomiting diarrhea stomach cramps',
        'nausea abdominal pain loose motions',
        'stomach pain vomiting fever',
        'chest pain burning sensation',
        'acidity heartburn after eating',
        'sneezing itchy eyes runny nose',
        'skin rash itching red spots',
        'hives swelling redness',
        'red rash itching skin irritation',
        'skin bumps redness inflammation',
        'wheezing shortness of breath chest tightness',
        'coughing difficulty breathing rapid breathing',
        'cough fever chest pain',
        'shortness of breath phlegm',
        'burning urination frequent urination cloudy urine',
        'painful urination bladder discomfort fever'
    ]
}

df = pd.DataFrame(data)
print(f"✅ Dataset: {len(df)} rows, {df['disease'].nunique()} diseases")
print(f"📊 Diseases: {df['disease'].unique()}")

# Train model
vectorizer = TfidfVectorizer(max_features=200)
X = vectorizer.fit_transform(df['symptoms'])
y = df['disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")

# Save
joblib.dump(model, 'models/disease_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
print("✅ Model saved!")

# Final test
print("\n🧪 FINAL TEST — These MUST work:")
test_cases = [
    ("fever chills headache", "Malaria or Fever"),
    ("runny nose sneezing", "Common Cold"),
    ("joint pain swelling", "Arthritis"),
    ("frequent urination thirst", "Diabetes"),
    ("headache chest pain dizziness", "Hypertension"),
    ("itching skin rash", "Allergy"),
    ("vomiting diarrhea", "Food Poisoning")
]

for symptoms, expected in test_cases:
    vec = vectorizer.transform([symptoms])
    pred = model.predict(vec)[0]
    status = "✅" if pred in expected else "⚠️"
    print(f"{status} '{symptoms}' → {pred} (Expected: {expected})")