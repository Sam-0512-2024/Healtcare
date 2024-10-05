import pandas as pd

# Load the dataset
file_path = 'E:\Healthcare\healthcare_reviews.csv'
df = pd.read_csv(file_path)

# Display basic information and the first few rows
print(df.info())  # Check columns, data types, and null values
print(df.head())  # Preview the first few rows of the dataset
import pandas as pd
import numpy as np

# Load the dataset again (already done previously)
# file_path = 'E:/Heathcare/healthcare_reviews.csv'
# df = pd.read_csv(file_path)

# 1. Drop rows with missing Review_Text
df.dropna(subset=['Review_Text'], inplace=True)

# 2. Map Ratings to Sentiment (Positive, Neutral, Negative)
def map_sentiment(rating):
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'

df['Sentiment'] = df['Rating'].apply(map_sentiment)

# 3. Clean the text in Review_Text (remove punctuation, convert to lowercase)
import re
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

df['Cleaned_Review_Text'] = df['Review_Text'].apply(clean_text)

# Display the updated dataset
print(df.head())
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Split the dataset into training and testing sets
X = df['Cleaned_Review_Text']  # Features (the cleaned review text)
y = df['Sentiment']  # Labels (sentiment: positive, neutral, negative)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Initialize the TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000)  # Limit to the top 5000 features

# 3. Fit the vectorizer on the training data and transform both training and testing data
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Check the shape of the transformed data
print(f'Training data shape: {X_train_tfidf.shape}')
print(f'Testing data shape: {X_test_tfidf.shape}')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# 2. Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# 3. Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# Re-train Logistic Regression with class weights
model = LogisticRegression(max_iter=200, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model again
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
from sklearn.svm import SVC

# Train an SVM classifier
svm_model = SVC(kernel='linear', class_weight='balanced')
svm_model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred_svm = svm_model.predict(X_test_tfidf)

# Evaluate the SVM model
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
from sklearn.model_selection import GridSearchCV

# Example grid for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'max_iter': [100, 200, 300]
}

grid = GridSearchCV(LogisticRegression(class_weight='balanced'), param_grid, cv=5)
grid.fit(X_train_tfidf, y_train)

# Evaluate the best model from grid search
best_model = grid.best_estimator_
y_pred_grid = best_model.predict(X_test_tfidf)

# Evaluate the grid search model
print("Grid Search Logistic Regression Report:\n", classification_report(y_test, y_pred_grid))
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data from your classification report
report = {
    'class': ['Negative', 'Neutral', 'Positive'],
    'precision': [0.41, 0.16, 0.40],
    'recall': [0.52, 0.40, 0.08],
    'f1-score': [0.46, 0.22, 0.13],
}

# Create a DataFrame
report_df = pd.DataFrame(report)

# Set the class as index for easier plotting
report_df.set_index('class', inplace=True)

# Visualization
plt.figure(figsize=(8, 6))

# Bar plots for Precision, Recall, and F1-score
bar_width = 0.25
x = range(len(report_df))

# Create bars
plt.bar(x, report_df['precision'], width=bar_width, label='Precision', color='blue', alpha=0.6)
plt.bar([p + bar_width for p in x], report_df['recall'], width=bar_width, label='Recall', color='orange', alpha=0.6)
plt.bar([p + bar_width * 2 for p in x], report_df['f1-score'], width=bar_width, label='F1-score', color='green', alpha=0.6)

# Adding labels and title
plt.xlabel('Classes', fontsize=14)
plt.ylabel('Scores', fontsize=14)
plt.title('Classification Report Metrics', fontsize=16)
plt.xticks([p + bar_width for p in x], report_df.index)
plt.ylim(0, 1)
plt.axhline(0, color='grey', lw=0.8)
plt.legend()

# Show plot
plt.tight_layout()
plt.show()
