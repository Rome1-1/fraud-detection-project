import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the training data
X_fraud_train = pd.read_csv('data/X_fraud_train.csv')
y_fraud_train = pd.read_csv('data/y_fraud_train.csv')

X_creditcard_train = pd.read_csv('data/X_creditcard_train.csv')
y_creditcard_train = pd.read_csv('data/y_creditcard_train.csv')

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'MLP': MLPClassifier(max_iter=1000)
}

# Train and evaluate models
def train_and_evaluate(X_train, y_train, models):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        results[name] = accuracy
    return results

# Evaluate on fraud data
fraud_results = train_and_evaluate(X_fraud_train, y_fraud_train, models)
print("Fraud Data Model Performance:\n", fraud_results)

# Evaluate on credit card data
creditcard_results = train_and_evaluate(X_creditcard_train, y_creditcard_train, models)
print("Credit Card Data Model Performance:\n", creditcard_results)
