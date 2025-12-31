import joblib
import pandas as pd

# Load trained model 
model = joblib.load("model/final_product_category_model.pkl")

print("Model loaded successfully!")
print("Type 'exit' to quit.\n")

while True:
    title = input("Enter product title: ")
    if title.lower() == "exit":
        print("Exiting prediction.")
        break

    # Prepare input
    data = pd.DataFrame({"title": [title.lower()]})

    # Predict
    prediction = model.predict(data)[0]

    print(f"Predicted category: {prediction}\n") 