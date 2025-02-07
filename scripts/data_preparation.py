import pandas as pd
from sklearn.model_selection import train_test_split

# Load your datasets
fraud_data = pd.read_csv('data/Fraud_Data_Cleaned.csv')
creditcard_data = pd.read_csv('data/CreditCard_Cleaned.csv')

# Separate features and target variable
X_fraud = fraud_data.drop(columns=['class'])
y_fraud = fraud_data['class']

X_creditcard = creditcard_data.drop(columns=['Class'])
y_creditcard = creditcard_data['Class']

# Train-test split for fraud data
X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(X_fraud, y_fraud, test_size=0.3, random_state=42)

# Train-test split for credit card data
X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test = train_test_split(X_creditcard, y_creditcard, test_size=0.3, random_state=42)

# Save the split data (optional)
X_fraud_train.to_csv('data/X_fraud_train.csv', index=False)
X_fraud_test.to_csv('data/X_fraud_test.csv', index=False)
y_fraud_train.to_csv('data/y_fraud_train.csv', index=False)
y_fraud_test.to_csv('data/y_fraud_test.csv', index=False)

X_creditcard_train.to_csv('data/X_creditcard_train.csv', index=False)
X_creditcard_test.to_csv('data/X_creditcard_test.csv', index=False)
y_creditcard_train.to_csv('data/y_creditcard_train.csv', index=False)
y_creditcard_test.to_csv('data/y_creditcard_test.csv', index=False)
