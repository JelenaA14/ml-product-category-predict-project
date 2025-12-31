import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Load dataset
df = pd.read_csv("data/products.csv")

# Rename columns for easier usage
df = df[["Product Title", " Category Label"]]
df.columns = ["title", "category"]

# Drop missing values
df = df.dropna(subset=["title", "category"])

# Convert product titles to lowercase
df["title"] = df["title"].str.lower()

# Clean and standardize category labels
df["category"] = (df["category"].str.strip().str.lower())

# Werge only clear singular/plural duplicates
category_mapping = {
    "fridge": "fridges",
    "fridges": "fridges",
    "cpu": "cpus",
    "cpus": "cpus",
    "mobile phone": "mobile phones",
    "mobile phones": "mobile phones"
}

df["category"] = df["category"].replace(category_mapping)

# Features and label
X = df[["title"]]
y = df["category"]


# Preprocessing and model pipelina
preprocessor = ColumnTransformer(transformers=[("text", TfidfVectorizer(stop_words="english", max_features=5000), "title")])

pipeline = Pipeline([("preprocessing", preprocessor), 
                     ("classifier", LinearSVC())])

# Train final model on all data
pipeline.fit(X, y)

# Save model
joblib.dump(pipeline, "model/final_product_category_model.pkl")

print("Model training completed and saved successfully")