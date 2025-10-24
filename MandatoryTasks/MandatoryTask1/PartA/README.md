
# TASK 1.A: WORD EMBEDDING

this task is to experiment and create word embeddings based on the given sentences and the context. 
i'll be using the popular Word2Vec: A core NLP technique.

Word2Vec is a method for creating vector embeddings of words that capture their semantic meanings. Developed by Tomas Mikolov and colleagues at Google in 2013





# Word2Vec

this task is to experiment and create word embeddings based on the given sentences and the context. 
i'll be using the popular Word2Vec: A core NLP technique.

Word2Vec is a method for creating vector embeddings of words that capture their semantic meanings. Developed by Tomas Mikolov and colleagues at Google in 2013


## 🚀 Project Overview

The notebook `word_embed.ipynb` builds a preprocessing and embedding workflow for textual data.  
It performs:
1. Data loading from Hugging Face Datasets  
2. Text cleaning and normalization  
3. Tokenization and stopword removal  
4. Lemmatization using POS tagging  
5. Training a Word2Vec model to generate semantic embeddings  


## 📦 Dependencies

the following libraries were used in this task

1. pandas and numpy: for dataframe handling and numerical operation
2. nltk: Tokenization, stopword removal, and Lemmatization
3. gensim: Training the Word2Vec word embeddings
4. Datasets: Loading the NewsQA dataset
5. ast: Parsing list strings back into Python lists

install it using the following cmd:

````markdown
pip install pandas numpy nltk gensim datasets
````

# 🪜PIPELINE STEPS

## 1. LOADING THE DATASET

•Source: lucadiliello/newsqa from Hugging Face Datasets
•


•Size:
Training set: 74,160 sample,
validation set: 4,212 samples


•Structure: Each sample contains context (news article), question, answer, key, and labels

````markdown
from datasets import load_dataset
ds = load_dataset("lucadiliello/newsqa") 
````

## 2. TEXT PREPROCESSING

### a. Text Combination
Combines context and question into a single text column for training. and also lower case all the letters
### b. Tokenization

Uses NLTK's word_tokenize for splitting text into tokens

### c. Text Cleaning

Removs stopwords using NLTK's English stopwords list
Additional custom stopwords removed: punctuation marks and special characters

### d. Lemmatization

Implements Part-of-Speech (POS) tagging for context-aware lemmatization

Uses WordNet Lemmatizer to reduce words to their base forms
Maps POS tags to appropriate WordNet categories (ADJ, VERB, NOUN, ADV)

## 3. Word2Vec Model Training

### Model Parameters:

• vector_size: 200 (dimensionality of word vectors)

• window: 5 (context window size)

• min_count: 3 (minimum word frequency threshold)

• sg: 0 (CBOW architecture, set to 1 for Skip-gram)

• epochs: 10 (number of training iterations)

• workers: CPU count - 1 (parallel processing)

### Training Features:

• Custom epoch logger callback for monitoring progress

• Compute loss tracking enabled

• Multi-core processing for faster training

### Output:

• Vocabulary Size: 94,925 unique words

• Saved Model: word2vec_model.model

• Embeddings CSV: top50_word_embeddings.csv (top 50 most frequent words)


## 4. Files Generated

1. tokenized_only.csv - Preprocessed tokenized text
2. word2vec_model.model - Trained Word2Vec model
3. top50_word_embeddings.csv - 200-dimensional embeddings for top 50 words

# 🗒️NOTES AND FUTURE IMPORVMENTS 

• The model uses CBOW architecture (faster, better for frequent words)

• For rare words or smaller datasets, consider switching to Skip-gram (sg=1)

• Preprocessing includes aggressive stopword removal and lemmatization for cleaner embeddings

