# FakeNewsDetection

# Overview

This project aims to classify news articles as either 'Real' or 'Fake' using machine learning. The code utilizes Python libraries such as NumPy, Pandas, and scikit-learn to preprocess the data and build a model based on the PassiveAggressiveClassifier.

# Features

Data preprocessing
TF-IDF vectorization
Custom list of stop words
PassiveAggressiveClassifier
Model evaluation using accuracy, confusion matrix, and classification report

# Requirements

Python 3.x
NumPy
Pandas
scikit-learn
Installation

To install the required packages, run the following command:

pip install numpy pandas scikit-learn

# How to Use

# Clone this repository.
Place your dataset named news.csv in the project directory. The dataset should have a 'label' column with values 'REAL' or 'FAKE', and a 'text' column containing the news article text.
Run main.py.

# Code Structure

Import Libraries: Importing necessary Python libraries.
Data Loading: Reading the data from a CSV file.
Data Exploration: Printing basic statistics and info about the dataset.
Data Preprocessing: Removing unnecessary columns.
Feature Extraction: Using TF-IDF vectorization.
Model Building: Using PassiveAggressiveClassifier.
Model Evaluation: Using accuracy, confusion matrix, and classification report.
Customization

You can add or remove stop words in the custom_stop_words list.
