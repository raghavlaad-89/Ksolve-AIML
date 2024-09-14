# Ksolve-AIML

# Aspect-Based Sentiment Analysis on Customer Reviews

## Overview

This project involves developing an aspect-based sentiment analysis model to process customer reviews and extract insights. The goal is to identify specific aspects of a business mentioned in the reviews and classify the sentiment associated with each aspect. The predefined aspects include Food Quality, Service, Ambiance, Pricing, and Cleanliness.

## Project Structure

1. **Data Preparation**: Unzip and load the dataset.
2. **Preprocessing**: Merge datasets, clean text, and prepare data.
3. **Aspect Extraction**: Identify aspects mentioned in reviews using a keyword-based approach.
4. **Sentiment Analysis**: Use TextBlob for sentiment analysis and BERT for advanced sentiment classification.
5. **Evaluation**: Measure the performance of aspect extraction and sentiment classification.
6. **Visualization**: Generate confusion matrices and other visualizations.

## Prerequisites

- Python 3.10 or higher
- Google Colab (or a similar Jupyter notebook environment)
- Libraries: `pandas`, `spacy`, `textblob`, `transformers`, `seaborn`, `matplotlib`, `sklearn`, `google.colab`

## Setup

1. **Mount Google Drive**: Ensure you have the dataset saved in your Google Drive.

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. **Unzip the Dataset**: Modify the paths if necessary.

    ```python
    import zipfile
    import os

    zip_path = '/content/drive/MyDrive/Colab Notebooks/Problem Statement 2/dataset.zip'
    extract_dir = '/content/dataset/dataset'

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    ```

## Code Execution

1. **Load and Preprocess Data**:

    ```python
    import pandas as pd

    def load_sample(file_path, nrows=100):
        return pd.read_json(file_path, lines=True, nrows=nrows)

    business_df = load_sample('/content/dataset/dataset/yelp_academic_dataset_business.json')
    review_df = load_sample('/content/dataset/dataset/yelp_academic_dataset_review.json')

    def preprocess_data(review_df, business_df):
        df = review_df.merge(business_df[['business_id', 'name', 'categories']], on='business_id', how='left')
        df['text'] = df['text'].apply(lambda x: x.lower())
        return df[['business_id', 'name', 'categories', 'stars', 'text']]

    df = preprocess_data(review_df, business_df)
    ```

2. **Aspect Extraction**:

    ```python
    import spacy

    nlp = spacy.load("en_core_web_sm")

    aspect_keywords = {
        'Food Quality': ['food', 'quality', 'taste', 'flavor', 'dish'],
        'Service': ['service', 'staff', 'waiter', 'waitress', 'customer service'],
        'Ambiance': ['ambiance', 'atmosphere', 'environment', 'decor'],
        'Pricing': ['price', 'cost', 'value', 'expensive', 'cheap'],
        'Cleanliness': ['cleanliness', 'clean', 'dirty', 'hygiene']
    }

    def extract_aspects(review):
        doc = nlp(review)
        aspects = []
        for aspect in aspect_keywords:
            if any(keyword in review for keyword in aspect_keywords[aspect]):
                aspects.append(aspect)
        return aspects

    df['aspects'] = df['text'].apply(extract_aspects)
    ```

3. **Sentiment Analysis**:

    ```python
    from textblob import TextBlob
    from transformers import pipeline, AutoTokenizer

    def get_sentiment(text):
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

    df['sentiment'] = df['text'].apply(get_sentiment)

    classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

    def bert_sentiment_analysis(text):
        if isinstance(text, str) and text.strip():
            inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
            sentiment = classifier(inputs['input_ids'])[0]
            return sentiment['label'].lower()
        else:
            return 'unknown'

    df['bert_sentiment'] = df['text'].apply(bert_sentiment_analysis)
    ```

4. **Evaluation**:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    def calculate_aspect_metrics(true_aspects, predicted_aspects):
        true_aspects_flat = [aspect for sublist in true_aspects for aspect in sublist]
        pred_aspects_flat = [aspect for sublist in predicted_aspects for aspect in sublist]

        precision = len(set(true_aspects_flat) & set(pred_aspects_flat)) / len(set(pred_aspects_flat)) if len(set(pred_aspects_flat)) > 0 else 0
        recall = len(set(true_aspects_flat) & set(pred_aspects_flat)) / len(set(true_aspects_flat)) if len(set(true_aspects_flat)) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    true_aspects = [['food', 'service'], ['location', 'ambiance'], ['staff', 'cleanliness']]
    pred_aspects = df['aspects'].tolist()[:3]

    precision, recall, f1 = calculate_aspect_metrics(true_aspects[0], pred_aspects)
    print(f"Aspect Extraction - Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

    y_true = test_df['stars'].apply(lambda x: 'positive' if x > 3 else 'negative' if x < 3 else 'neutral')
    y_pred = test_df['sentiment']

    accuracy = accuracy_score(y_true, y_pred)
    precision_sentiment = precision_score(y_true, y_pred, average='weighted', labels=['positive', 'neutral', 'negative'])
    recall_sentiment = recall_score(y_true, y_pred, average='weighted', labels=['positive', 'neutral', 'negative'])
    f1_sentiment = f1_score(y_true, y_pred, average='weighted', labels=['positive', 'neutral', 'negative'])

    print(f"Sentiment Classification - Accuracy: {accuracy:.2f}, Precision: {precision_sentiment:.2f}, Recall: {recall_sentiment:.2f}, F1 Score: {f1_sentiment:.2f}")

    conf_matrix = confusion_matrix(y_true, y_pred, labels=['positive', 'neutral', 'negative'])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['positive', 'neutral', 'negative'], yticklabels=['positive', 'neutral', 'negative'])
    plt.title('Confusion Matrix for Sentiment Analysis')
    plt.xlabel('Predicted Sentiment')
    plt.ylabel('True Sentiment')
    plt.show()

    bert_y_pred = test_df['bert_sentiment']
    bert_accuracy = accuracy_score(y_true, bert_y_pred)
    bert_f1_sentiment = f1_score(y_true, bert_y_pred, average='weighted', labels=['positive', 'neutral', 'negative'])

    print(f"BERT Sentiment Classification - Accuracy: {bert_accuracy:.2f}, F1 Score: {bert_f1_sentiment:.2f}")
    ```

## Results

The output includes:
- **Aspect Extraction Metrics**: Precision, Recall, and F1 Score for aspect extraction.
- **Sentiment Classification Metrics**: Accuracy, Precision, Recall, and F1 Score for both TextBlob and BERT sentiment analysis.
- **Confusion Matrix**: Visualization of sentiment classification performance.


## Running the Code

1. **Set Up**: Ensure you have the necessary libraries and data.
2. **Run**: Execute the code in a Jupyter notebook or Google Colab environment.
3. **Evaluate**: Review the printed metrics and visualizations to understand the model's performance.


