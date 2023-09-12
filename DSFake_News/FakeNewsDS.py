#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Custom list of stop words
custom_stop_words = ['hillary', 'watch', 'kerry', 'bernie', 'battle', 'new', 'york']
all_stop_words = list(ENGLISH_STOP_WORDS.union(custom_stop_words)) # Converted to a list.

# Read the data
df = pd.read_csv('/Users/brocktbennett/GitHub/FakeNewsDetection/DSFake_News/news.csv')

# Drop unnecessary columns
df.drop(['Unnamed: 0'], axis=1, inplace=True)

# Data Exploration
print("Shape of the dataset:", df.shape)
print("\nFirst 5 records:\n", df.head())
print("\nLabel Distribution:\n", df['label'].value_counts())
print("\nChecking for missing values:\n", df.isnull().sum())

# Displaying words removed from the first 5 records
for index, row in df.head().iterrows():
    original_words = set(row['text'].split())
    words_after_stop_words = original_words.difference(all_stop_words)
    removed_words = original_words.difference(words_after_stop_words)
    print(f"\nRemoved stop words from record {index}: {removed_words}")

# Get the labels
labels = df.label

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

# Initialize a TfidfVectorizer with custom stop words
tfidf_vectorizer = TfidfVectorizer(stop_words=all_stop_words, max_df=0.7)

# Fit and transform the train set, transform the test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# Initialize PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)

# Cross-Validation for PAC
pac_cv_scores = cross_val_score(pac, tfidf_train, y_train, cv=5)
print("\nCross-validation scores for PAC:", pac_cv_scores)
print("Mean cross-validation score for PAC:", np.mean(pac_cv_scores))

# Fit the model
pac.fit(tfidf_train, y_train)

# Predict and evaluate
y_pred = pac.predict(tfidf_test)
print("\nAccuracy Score:", round(accuracy_score(y_test, y_pred) * 100, 2))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))
