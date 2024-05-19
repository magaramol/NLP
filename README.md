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


# Word2Vec README

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







## Usage

To train a Word2Vec model, you can use popular libraries such as Gensim in Python. Below is an example of how to train a Word2Vec model using Gensim:
```


# Conclusion

Word2Vec is a powerful technique for creating word embeddings that capture semantic meaning and relationships between words. By choosing between CBOW and Skip-Gram models based on the dataset size, you can optimize the performance and efficiency of your natural language processing tasks.
