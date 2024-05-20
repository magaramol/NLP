## Natural Language Processing

### Task Description

The goal of this task is to preprocess text data for analysis or modeling for NLP tasks. The following steps are applied to the text:

1. Convert all text to lowercase.
2. Remove HTML tags.  
3. Remove URLs.
4. Remove punctuation.
5. Implement chat word treatment using the SMS slang translator available at [GitHub](https://github.com/rishabhverma17/sms_slang_translator).
6. Perform spelling correction.
7. Remove stopwords.
8. Handle emojis.

### Data Sources


1) IMDB_ Dataset.csv: [Download from GitHub](https://github.com/magaramol/NLP/blob/main/OMDB_data/IMDB_%20Dataset.csv) (Source: [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews))
2) movies.csv: [Download from GitHub](https://github.com/magaramol/NLP/blob/main/OMDB_data/movies.csv)

Data fetched from OMDB API: [OMDB API](https://www.omdbapi.com/)


# Word2Vec

Word2Vec is a popular neural network-based model used for generating word embeddings, where words are mapped to vectors of real numbers in a continuous vector space. This README covers the advantages of Word2Vec, the types of Word2Vec models, and the differences between Continuous Bag of Words (CBOW) and Skip-Gram.

## Advantages of Word2Vec

1. **Capture Semantic Meaning**: Word2Vec captures the semantic relationships between words. Words with similar meanings are positioned closely in the vector space, enabling the model to understand context and similarity.
2. **Low Dimension**: Word2Vec generates low-dimensional vectors, which reduce computational complexity and improve the efficiency of downstream tasks.
3. **Dense Vectors**: The vectors produced are dense, meaning they have fewer zero values. This helps in tackling overfitting issues, as it allows the model to generalize better from the training data.

## Types of Word2Vec Models

### Continuous Bag of Words (CBOW)

The CBOW model predicts the target word (center word) using the context words (surrounding words). This model is effective for smaller datasets and tends to converge faster due to the averaging of context words. It works by averaging the context word vectors and using this average to predict the target word.

**Characteristics of CBOW**:
- Suitable for small datasets.
- Faster training due to context averaging.
- Predicts a word based on its context.

### Skip-Gram

The Skip-Gram model does the opposite of CBOW. It predicts the context words from the target word (center word). Skip-Gram is more suitable for larger datasets and can capture a wider range of word relationships. It works by using the target word to predict the probability of each context word.

**Characteristics of Skip-Gram**:
- Suitable for large datasets.
- Slower training but captures more detailed word relationships.
- Predicts context words based on a single word.

## Usage

To train a Word2Vec model, you can use popular libraries such as Gensim in Python. Below is an example of how to train a Word2Vec model using Gensim:

```python
from gensim.models import Word2Vec

# Example sentences
sentences = [
    ['word1', 'word2', 'word3'],
    ['word4', 'word5', 'word6'],
    # Add more sentences here
]

# Training the Word2Vec model using CBOW
model_cbow = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=0)

# Training the Word2Vec model using Skip-Gram
model_skipgram = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1)

# Save the model
model_cbow.save("word2vec_cbow.model")
model_skipgram.save("word2vec_skipgram.model")

# Load the model
model_cbow = Word2Vec.load("word2vec_cbow.model")
model_skipgram = Word2Vec.load("word2vec_skipgram.model")


```


# Conclusion

Word2Vec is a powerful technique for creating word embeddings that capture semantic meaning and relationships between words. By choosing between CBOW and Skip-Gram models based on the dataset size, you can optimize the performance and efficiency of your natural language processing tasks.


# Text Classification

## Types of Classification in Machine Learning (Supervised Learning)

1. **Tabular**
    - Involves structured data with rows and columns, similar to a spreadsheet or database table.
    - Common tasks include regression (predicting continuous values) and classification (assigning categories to instances).
    - Examples: Predicting house prices, customer churn prediction.

2. **Image Classification**
    - Involves classifying images into predefined categories.
    - Uses convolutional neural networks (CNNs) and other deep learning techniques.
    - Examples: Identifying objects in images, facial recognition.

3. **Text Classification**
    - Involves categorizing text into predefined categories.
    - Uses natural language processing (NLP) and machine learning techniques.
    - Examples:
        - Email spam or ham detection
        - Sentiment analysis
        - Message categorization (e.g., sales vs. support)

## Types of Text Classification

1. **Binary Classification**: Two classes
    - The task is to classify instances into one of two possible categories.
    - Examples: Spam vs. ham emails, positive vs. negative sentiment.

2. **Multiclass Classification**: More than two classes
    - The task is to classify instances into one of three or more possible categories.
    - Examples: Classifying news articles into sports, politics, entertainment.

3. **Multilabel Classification**: Instances can belong to multiple classes
    - Each instance can be assigned multiple labels.
    - Examples: Classifying a news article as both sports and politics.

## Applications

1. **Email Filtering**
    - Determine if an email is spam or ham.
    - Example: Gmail's spam filter.

2. **Sentiment Analysis**
    - Analyze sentiment in text data, such as reviews or social media posts.
    - Example: Determining if a product review is positive, negative, or neutral.

3. **Language Detection**
    - Identify the language of a given text.
    - Example: Google Translate detecting the input language.

4. **Fake News Detection**
    - Identify and classify news articles as real or fake.
    - Uses NLP and machine learning techniques to analyze the content and metadata of news articles.
    - Example: Platforms like Factmata and Fake News Challenge.

## Pipeline

1. **Data Acquisition**
   - Import data from various sources, such as CSV files, databases, or APIs.
   - Example: Using pandas to load a dataset from a CSV file.

2. **Text Preprocessing**
   - Clean and preprocess text data to make it suitable for analysis.
   - Steps include:
     - Converting text to lowercase
     - Removing HTML tags
     - Removing URLs
     - Removing punctuation
     - Implementing chat word treatment
     - Spelling correction
     - Removing stopwords
     - Handling emojis

3. **Text Vectorization**
   - Convert text to numerical vectors using methods such as:
     - Bag of Words (BoW)
     - TF-IDF (Term Frequency-Inverse Document Frequency)
     - Word2Vec

4. **Modeling**
   - Build and train models to classify text.
   - Machine Learning models: Naive Bayes, Random Forest
   - Deep Learning models: RNN, LSTM, CNN, BERT

5. **Evaluation**
   - Use metrics to evaluate model performance.
   - Common metrics:
     - Accuracy: Proportion of correctly classified instances
     - Confusion Matrix: Table showing true vs. predicted classifications

6. **Model Deployment**
   - Deploy the model to a production environment.
   - Platforms include:
     - AWS (Amazon Web Services)
     - GCP (Google Cloud Platform)
     - Azure (Microsoft Azure)

### Steps in Detail

1. **Data Acquisition**
   - Import data from various sources such as CSV files, databases, or APIs.
   - Example: Using pandas to load a dataset from a CSV file.

2. **Text Preprocessing**
   - Clean and preprocess text data to make it suitable for analysis.
   - Steps include converting text to lowercase, removing HTML tags, URLs, punctuation, implementing chat word treatment, spelling correction, removing stopwords, and handling emojis.

3. **Text Vectorization**
   - Convert text to numerical vectors using methods such as Bag of Words (BoW), TF-IDF, and Word2Vec.

4. **Modeling**
   - Build and train models to classify text.
   - Machine Learning models: Naive Bayes, Random Forest
   - Deep Learning models: RNN, LSTM, CNN, BERT

5. **Evaluation**
   - Use metrics to evaluate model performance.
   - Common metrics: Accuracy, Confusion Matrix

6. **Model Deployment**
   - Deploy the model to a production environment.
   - Platforms include AWS (Amazon Web Services), GCP (Google Cloud Platform), and Azure (Microsoft Azure).

## Different Approaches

1. **Heuristic Approach**
   - Useful when dealing with limited data.
   - Involves using domain knowledge to manually create rules for classification.

2. **API-Based Approach**
   - Utilize commercial APIs (e.g., AWS, GCP, NLP Cloud).
   - Example: [NLP Cloud Sentiment Analysis](https://nlpcloud.com/home/playground/sentiment-analysis)
   - Advantages: No need to develop models from scratch
   - Disadvantages: Can incur costs

3. **Machine Learning Approach**
   - Techniques:
     - Bag of Words (BoW)
     - TF-IDF (for information retrieval)
     - Word2Vec (pretrained or custom-trained on your data)

4. **Deep Learning Approach**
   - Models: RNN, LSTM, CNN, BERT

## Final Words

1. Use ensemble models for better performance.
2. Incorporate heuristic features where applicable.
3. Start with machine learning models before diving into deep learning.
4. Handle imbalanced datasets carefully.
5. Focus on practical projects to gain hands-on experience.
