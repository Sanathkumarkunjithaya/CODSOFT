# SPAM SMS DETECTION

## Overview
This repository contains an implementation of an SMS spam detection system using machine learning techniques. The project focuses on building an AI model that classifies SMS messages as either spam or legitimate. The implemented solution employs techniques such as TF-IDF (Term Frequency-Inverse Document Frequency) or word embeddings, coupled with classifiers like Naive Bayes, Logistic Regression, or Support Vector Machines.

## Dataset
The dataset used for training and testing the model can be found on Kaggle: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

## Code Structure
The code is written in Python and utilizes Jupyter Notebook. The main components of the code are as follows:

### 1. Importing Dependencies
- Necessary libraries such as NumPy, Pandas, and scikit-learn are imported.

### 2. Data Collection & Pre-Processing
- The SMS data is loaded from a CSV file into a Pandas DataFrame.
- Null values are replaced with an empty string.
- The data is labeled: spam messages as 0 and ham (legitimate) messages as 1.

### 3. Feature Extraction
- Text data is transformed into feature vectors using TF-IDF Vectorizer.
- The dataset is split into training and testing sets.

### 4. Training the Model
- Logistic Regression is chosen as the classifier.
- The model is trained using the training data.
- Accuracy scores on both training and test data are calculated.

### 5. Building a Predictive System
- A simple predictive system is demonstrated using an example input SMS message.
- The input is converted to feature vectors, and the model predicts whether it's spam or ham.
- The result is printed to the console.

## How to Run
1. Clone the repository to your local machine.
2. Download the dataset from the provided Kaggle link and place it in the project directory.
3. Open and run the Jupyter Notebook in a Python environment, ensuring all dependencies are installed.

Feel free to explore and modify the code according to your requirements.

## Author
[Sanath Kumar Kunjithaya]

## Acknowledgments
- This project was completed as part of an internship at [Codsoft].
- Special thanks to the contributors and maintainers of the scikit-learn library.

