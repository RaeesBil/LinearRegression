import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('LR.csv')

print("First few rows of the dataset:\n", df.head())
print("\nColumn names:\n", df.columns)

X = df.drop('charges', axis=1)
y = df['charges']

numeric_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

model = Pipeline(steps=[('preprocessor', preprocessor),
                         ('regressor', LinearRegression())])

param_grid = {
    'regressor__fit_intercept': [True, False],
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

grid_search.fit(X_train, y_train)

print("Best parameters found:\n", grid_search.best_params_)

y_pred_best = grid_search.best_estimator_.predict(X_test)

print("Best Model MAE:", mean_absolute_error(y_test, y_pred_best))
print("Best Model MSE:", mean_squared_error(y_test, y_pred_best))
print("Best Model R² Score:", r2_score(y_test, y_pred_best))

residuals = y_test - y_pred_best

plt.figure(figsize=(10, 6))
plt.scatter(y_pred_best, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

cv_scores = cross_val_score(grid_search.best_estimator_, X, y, cv=5, scoring='neg_mean_squared_error')
print("Cross-Validation Scores (Negative MSE):", cv_scores)
print("Mean CV MSE:", -cv_scores.mean())