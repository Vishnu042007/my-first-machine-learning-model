# ==========================================
# STEP 1: IMPORT INDUSTRY STANDARD LIBRARIES
# ==========================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# STEP 2: LOAD AND INSPECT THE DATA
# ==========================================
# Load the CSV file containing house data (Area, Bedrooms, Price)
df = pd.read_csv('data.csv')

# Separate features (X) from the target price we want to predict (y)
X = df[['Area_SqFt', 'Bedrooms']]
y = df['Price_Lakhs']

# ==========================================
# STEP 3: SPLIT DATA FOR TRAINING & TESTING
# ==========================================
# Use 80% of data to train the model and save 20% to test its accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# STEP 4: TRAIN THE LINEAR REGRESSION MODEL
# ==========================================
# Initialize the classical mathematical model
model = LinearRegression()

# Train the model by showing it the training data points
model.fit(X_train, y_train)

# ==========================================
# STEP 5: EVALUATE HOW ACCURATE IT IS
# ==========================================
# Make predictions on the test dataset
predictions = model.predict(X_test)

# Calculate error metrics
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Model Training Complete!")
print(f"Mean Absolute Error: ₹{mae:.2f} Lakhs")
print(f"R-squared Score (Accuracy metric): {r2:.2f}")
