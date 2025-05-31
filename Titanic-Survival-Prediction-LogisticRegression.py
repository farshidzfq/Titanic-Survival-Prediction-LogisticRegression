import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set seaborn style for plots
sns.set(style="whitegrid")

# 1. Load dataset
df = pd.read_csv('titanic.csv')

# Display first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# 2. Check for missing values and dataset info
print("\nDataset info:")
print(df.info())

print("\nNumber of missing values in each column:")
print(df.isnull().sum())

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

print("\nMissing values after imputation:")
print(df.isnull().sum())

# 3. Convert categorical columns to numeric

# Map 'Sex': male=1, female=0
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

# One-hot encode 'Embarked'
embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
df = pd.concat([df, embarked_dummies], axis=1)
df.drop('Embarked', axis=1, inplace=True)

# 4. Feature engineering

# Create 'FamilySize' feature
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Categorize 'Age' into bins and one-hot encode
bins = [0, 12, 20, 40, 60, 120]
labels = ['Child', 'Teenager', 'Adult', 'Senior', 'Elderly']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
age_dummies = pd.get_dummies(df['AgeGroup'], prefix='AgeGroup')
df = pd.concat([df, age_dummies], axis=1)
df.drop('AgeGroup', axis=1, inplace=True)

# 5. Exploratory Data Analysis (optional, can comment out if running frequently)
"""
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title('Survival Counts')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.ylabel('Survival Rate')
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()

plt.figure(figsize=(8,4))
sns.barplot(x='FamilySize', y='Survived', data=df)
plt.title('Survival Rate by Family Size')
plt.xlabel('Family Size')
plt.ylabel('Survival Rate')
plt.show()
"""

# 6. Prepare features and target variable
feature_cols = ['Pclass', 'Sex', 'Fare', 'FamilySize'] + list(embarked_dummies.columns) + list(age_dummies.columns)
X = df[feature_cols]
y = df['Survived']

# 7. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8. Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 9. Hyperparameter tuning with GridSearchCV for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [1000]
}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters from GridSearchCV: {grid_search.best_params_}")

best_model = grid_search.best_estimator_

# 10. Cross-validation accuracy on training set
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
print(f"Cross-validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")

# 11. Evaluate on test set
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Set Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted No', 'Predicted Yes'],
            yticklabels=['Actual No', 'Actual Yes'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 12. (Optional) Save the trained model and scaler for future use
"""
import joblib
joblib.dump(best_model, 'logistic_regression_titanic_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
"""

