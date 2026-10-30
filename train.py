import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

print("Credit Card Dataset load ho raha hai...")
df = pd.read_csv('creditcard.csv')

X = df.drop('Class', axis=1)
y = df['Class']

# Stratified split taaki balanced training ho
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4 Alag-alag Machine Learning Models
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42, n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

trained_models = {}

for name, model in models.items():
    print(f"Training -> {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model

# Models save karna
joblib.dump(trained_models, 'fraud_models_dict.pkl')
print("Sabhi 4 models successfully train aur save ho gaye hain!")