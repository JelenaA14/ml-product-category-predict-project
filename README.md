# 🛒 Product Category Prediction ML Project (Complete Pipeline)

This repository contains a complete machine learning pipeline for predicting product categories based on product titles using Python and scikit-learn.

The project was developed as part of a learning model, demonstrating all major phases of a real-world machine learning workflow - from raw data analysis to a trained and deployable model.


## 📁 Project Structure


├── data/
│   └── products.csv                
│
├── notebooks/
│   └── product_category_analysis.ipynb      
│
├── models/
│   └── final_product_category_model.pkl
│
├── src/
│   ├── train_model.py          
│   └── predict_category.py         
|
└── README.md


## ✅ What wWe Did in This Module

Throughout this project, we covered all essential steps of a real-world machine learning task:

### 1️⃣ Project Setup 

- Create a GitHub repository
- Defined a clear project folder structure
- Uploadea the raw dataset

### 2️⃣ Data Exploration

- Loaded and inspected the dataset
- Analyzed basic dataset properties (shape, columns, data types)
- Checked for missing values
- Explored the distribution of product categories


### 3️⃣ Data Cleaning & Preprocesing

- Removed rows with missing values
- Standardized text data ( lowercasing product titles)
- Clean and normalized category labels 
- Merged only clear singular/plural duplicates
- Ensured consistent category representation


### 4️⃣ Data Visualization

- Created a bar chart to visualize the distribution of product categories
- Identified class imbalance and dominant categories
- Used visualization results to better understand dataset structure
- Confirmed the need for text-based classification
- Visualizations were implemented as part of the exploratory data analysis using 'matplotlib'


### 5️⃣ Feature Engineering

- Selected **product titles** as the main input feature 
- Justified the exclusion of additional numeric features
- Focused on text-based representation for classification

### 6️⃣ Text Vectorization

- Applied **TF-IDF 
- Converted product titles into numerical feature vectors
- Reduced noise by removing common stop words

### 7️⃣ Model Training & Evaluation

- Trained and evaluated multiple machine learning models:
  - Logistic Regression
  - Naive Bayes
  - Decision Tree
  - Random Forest
  - Support Vector Machine
- Compared models using classification metrics
- Selected **Linear SVC** as the best-performing model

### 8️⃣ Final Model Treining

- Trained the final **Linear SVC** model on the full dataset
- Used a unified **Pipeline** for preprocessing and classification
- Saved the trained model for later use

### 9️⃣ Model Deployment & Prediction

- Implemented a standalone training script ('train_model.py')
- Implemented an interactive prediction script ('predict_model.py')
- Enabled real-time predictions using user input

## 🚀 How to Run the Project

###  Train the Final Modek

## 🧠 Conclusion

This project demonstrates an end-to-end machine learning pipeline for text classification. By combining TF-IDF vectorization
with a Linear Support Vector Machine, the model achieves strong performance while maintaining a clean and interpretable design,
The separation betweeen experimentation (notebook) and deployment (Python scripts) reflects best practices used in real-world machine learning projects.