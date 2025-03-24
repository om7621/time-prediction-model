import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load the larger synthetic dataset
data = pd.read_csv("synthetic_large_data.csv")

# Separate features and target
X = data[['product_category', 'customer_location', 'shipping_method']]
y = data['delivery_time']

# Identify categorical features
categorical_features = ['product_category', 'customer_location', 'shipping_method']

# Create preprocessing pipelines for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

# Create a pipeline with the preprocessor and a model
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('regressor', RandomForestRegressor(random_state=42))])

# Train the model
model.fit(X, y)

# Save the trained model
model_filename = 'delivery_time_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Trained model saved as {model_filename}")
