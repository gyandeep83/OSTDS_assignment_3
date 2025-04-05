import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load Dataset
df = pd.read_csv('/Users/gyandeep/OSTDS_assign_3/Datasets/diverse_salary_data.csv')

# Debugging: Print dataset columns to verify correctness
print("Columns in dataset:", df.columns)
print(df.head())

# Updated Feature Lists
categorical_features = ['Education', 'Location', 'Job_Title']
numerical_features = ['Experience', 'Age']

# Preprocessing
ohe = OneHotEncoder(drop='first', handle_unknown='ignore')
scaler = StandardScaler()

preprocessor = ColumnTransformer([
    ('num', scaler, numerical_features),
    ('cat', ohe, categorical_features)
])

# Train-Test Split
X = df.drop(columns=['Salary'])
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform data to check encoding correctness
X_train_transformed = preprocessor.fit_transform(X_train)
print("Transformed X shape:", X_train_transformed.shape)

# Models (only Ridge and Lasso)
models = {
    'Ridge Regression': Ridge(alpha=10),
    'Lasso Regression': Lasso(alpha=2.0)
}

results = {}

# Scatter Plot Setup
plt.figure(figsize=(12, 5))
for idx, (name, model) in enumerate(models.items(), start=1):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R²': r2}
    
    print(f"{name}: MSE = {mse:.2f}, R² = {r2:.2f}")
    
    # Scatter Plot - Actual vs Predicted Salaries
    plt.subplot(1, 2, idx)
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Actual Salary')
    plt.ylabel('Predicted Salary')
    plt.title(f'{name} - Actual vs Predicted')

plt.tight_layout()
plt.show()

# Bar Plot - MSE Comparison
plt.figure(figsize=(7, 5))
sns.barplot(x=list(results.keys()), y=[res['MSE'] for res in results.values()])
plt.ylabel('Mean Squared Error')
plt.title('Model Comparison (MSE)')
plt.show()
