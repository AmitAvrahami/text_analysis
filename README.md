# Assignment 3 - Text Analysis

## Overview
This project focuses on **text analysis** using Python, specifically analyzing a dataset of annotated stories. The primary goal is to classify these stories based on gender using various machine learning models. The project includes several key steps: text data preprocessing, feature extraction, model training, and evaluation.

## Data
The dataset includes two CSV files:
- **`annotated_corpus_for_train.csv`**: Contains training data with stories and their associated gender labels.
- **`corpus_for_test.csv`**: Contains test data with stories for which predictions will be made.

## Key Steps and Methods

### 1. Data Preprocessing
- **Cleaning Missing Values**: Rows with missing values are removed to ensure data integrity.
- **Removing Duplicates**: Duplicate entries are eliminated to avoid redundancy.

   ![Data Cleaning](https://github.com/user-attachments/assets/1c7e90f6-7395-4e1c-8942-9e7ae65b910e)
   *Example of data before and after cleaning.*

### 2. Text Tokenization
- A tokenizer splits the text into individual words, which facilitates text analysis and classification.

   ![Tokenization Example](https://github.com/user-attachments/assets/f915c604-02b6-472e-b360-f6ab589b6710)
   *Illustration of how text is tokenized.*

### 3. Feature Extraction
- **Count Vectorizer**: Converts text into a matrix of token counts, where each word in the stories becomes a feature.
- **TF-IDF Vectorizer**: Converts text into a matrix of TF-IDF scores, reflecting the importance of each word in the context of the document and corpus.

   ![Count Vectorizer](https://github.com/user-attachments/assets/98467708-26cc-49b0-813b-38c1112ef3c7)
   ![TF-IDF Vectorizer](https://github.com/user-attachments/assets/6a9e6c0a-9e1f-4206-aba6-0fc377186ec2)
   *Examples of feature extraction using Count and TF-IDF Vectorizers.*

### 4. Model Training
- Various machine learning models are employed, including Naive Bayes, K-Nearest Neighbors (KNN), and Decision Trees.
- A pipeline is used to combine the tokenizer with the chosen model, simplifying the training process.

   ![Model Training](https://github.com/user-attachments/assets/8431694a-0988-41cf-867f-d4e71c17e0ea)
   *Example of the model training process.*

### 5. Model Evaluation
- Cross-validation is performed to assess the performance of each model, using metrics like F1 score.

   ![Model Evaluation](https://github.com/user-attachments/assets/ad0ca281-b86e-4cd7-bcf0-0af2476107ac)
   *Model performance evaluation through cross-validation.*

### 6. Stop Words Removal
- Low-importance words, based on their TF-IDF scores, are removed to improve model performance.

   ![Stop Words Removal](https://github.com/user-attachments/assets/b655b446-5755-44d2-9b39-af6ad5cb8cee)
   *Illustration of stop words removal.*

### 7. Hyperparameter Tuning
- Grid Search is used to find the optimal parameters for each model, enhancing their accuracy.

   ![Hyperparameter Tuning](https://github.com/user-attachments/assets/3f304c33-2cca-4d03-9b61-6b77fc306bcb)
   *Example of hyperparameter tuning using Grid Search.*

### 8. Predictions and Results
- The best-performing models are used to predict genders for the test dataset.
- Results are compiled into a DataFrame for comparison.

   ![Predictions and Results](https://github.com/user-attachments/assets/f8a93ff9-3f25-4375-a74a-0387f10a53bd)
   *Predicted results for the test dataset.*

## Output
The predicted results are saved to a CSV file, including the test example ID and the predicted gender for each story.

   ![Output](https://github.com/user-attachments/assets/752dc028-486f-4382-b6fd-4e704ff68fce)
   *Example of the output CSV file.*

## Visualizations
The project includes visualizations to illustrate the data distribution and model performance. These visualizations are created using libraries such as Seaborn and Matplotlib.

## Libraries and Tools Used
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical operations.
- **Scikit-Learn**: Machine learning models and feature extraction.
- **Hebrew Tokenizer**: Tokenization for Hebrew text.
- **Matplotlib & Seaborn**: Visualization.

## Installation
To run this project, install the required Python packages using the following command:
```bash
pip install pandas numpy scikit-learn hebrew_tokenizer matplotlib seaborn
