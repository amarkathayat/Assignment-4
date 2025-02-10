import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('weight-height.csv')

# Display basic information
print(df.info())
print(df.head())

# Scatter plot to visualize the relationship
plt.scatter(df['Height'], df['Weight'], alpha=0.5)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Scatter Plot of Height vs Weight')
plt.show()
from sklearn.linear_model import LinearRegression

# Define X (independent) and y (dependent)
X = df[['Height']]  # Reshape X into a 2D array
y = df['Weight']

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Get predictions
y_pred = model.predict(X)
plt.scatter(df['Height'], df['Weight'], alpha=0.5, label='Actual Data')
plt.plot(df['Height'], y_pred, color='red', label='Regression Line')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Regression Line Over Scatter Plot')
plt.legend()
plt.show()
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f'RMSE: {rmse:.2f}')
print(f'R²: {r2:.4f}')
