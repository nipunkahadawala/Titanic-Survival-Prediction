# Titanic - Machine Learning from Disaster

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


def preprocess_data(df):
    
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    
    
    features = ['Pclass', 'Sex', 'Age', 'Fare']
    return df[features]


X = preprocess_data(train_data)
y = train_data['Survived']  
X_test = preprocess_data(test_data)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


val_score = model.score(X_val, y_val)
print(f"Validation Accuracy: {val_score:.4f}")


predictions = model.predict(X_test)


submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions
})
submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully!")