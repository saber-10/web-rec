
# TASK 1.B: LANGUAGE TRANSLATION(ENGLISH TO FRENCH)

this task is to experiment and create a Language Translation Model.
i have fine-tuned a MarianMT neural machine translation model for English-to-French translation using the OPUS Books dataset. The model is based on the Helsinki-NLP pre-trained transformer architecture and is optimized for book-style translations.







# Helsinki Transformer

i have fine-tuned a MarianMT neural machine translation model for English-to-French translation using the OPUS Books dataset. The model is based on the Helsinki-NLP pre-trained transformer architecture and is optimized for book-style translations.

Their most well-known project is OPUS-MT — a family of open-source translation models covering over 1,000 language pairs.


## 🚀 Project Overview

The notebook english_to_french.ipynb builds a complete translation pipeline for English-French language pairs.
It performs:

1. Data loading from Hugging Face Datasets
2. Train-test split (80-20)
3. Tokenization with SentencePiece
4. Sequence-to-sequence model fine-tuning
5. Translation generation with beam search
6. BLEU score evaluation for quality assessment  


## 📦 Dependencies

The following libraries were used in this task:

• transformers: For MarianMT model and training utilities

• datasets: Loading the OPUS Books dataset from Hugging Face

• torch: PyTorch backend for model training

• pandas and numpy: For data manipulation and numerical operations

• evaluate: BLEU score computation for translation quality

• wandb: Experiment tracking and monitoring

Install it using the following cmd:
````markdown
pip install transformers datasets torch pandas numpy evaluate wandb
````


# 🪜PIPELINE STEPS

## 1. LOADING THE DATASET

• Source: Helsinki-NLP/opus_books (en-fr) from Hugging Face Datasets
Domain: Literary texts from books
Size:

• Total samples: 127,085 parallel sentence pairs
Training set: 101,668 samples (80%)
Validation set: 25,417 samples (20%)


• Structure: Each sample contains id and translation dict with 'en' and 'fr' keys

````markdown
from datasets import load_dataset
dataset = load_dataset("Helsinki-NLP/opus_books", "en-fr")
````

## 2. DATA PREPARATION

### a. Train-Test Split
Creates 80-20 split for training and validation sets
### b. Tokenization
• Uses MarianTokenizer with SentencePiece encoding

• Handles both source (English) and target (French) sequences

from transformers import MarianTokenizer
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
````markdown
def tokenize(sentences):
    input = [x['en'] for x in sentences['translation']]
    translated = [x['fr'] for x in sentences['translation']]
    
    model_input = tokenizer(
        input, 
        truncation=True, 
        padding=True, 
        max_length=128,
        text_target=translated
    )
    return model_input
````

### c. Data Collation

Uses DataCollatorForSeq2Seq for dynamic padding and label preparation

Automatically handles padding masks and attention masks


## 3. MODEL FINE-TUNING

### Base Model:

• Architecture: MarianMT (Transformer encoder-decoder)

• Pre-trained: Helsinki-NLP/opus-mt-en-fr

• Parameters: ~74M parameters

• Tokenizer: SentencePiece with en-fr specialized vocabulary

### Training Parameters:

• learning_rate: 2e-5 (AdamW optimizer)

• per_device_train_batch_size: 16

• per_device_eval_batch_size: 16

• num_train_epochs: 3

• weight_decay: 0.01 (L2 regularization)

• fp16: True (mixed precision training)

• save_total_limit: 2 (keeps only 2 best checkpoints)

• predict_with_generate: True 


### MAKING PREDICTIONS
````markdown
text = "recruitments for web club will be on 26th october."
inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)
translated_tokens = model.generate(**inputs)
translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

print(translation)

# Output: "Les recrutements pour le web-club auront lieu le 26 octobre."
````


## 4. Files Generated

1. tokenized_only.csv - Preprocessed tokenized text
2. word2vec_model.model - Trained Word2Vec model
3. top50_word_embeddings.csv - 200-dimensional embeddings for top 50 words

# 🚀 USAGE 
````markdown
from transformers import MarianMTModel, MarianTokenizer

# Load model
model = MarianMTModel.from_pretrained("./en-fr-translation-model")
tokenizer = MarianTokenizer.from_pretrained("./en-fr-translation-model")

# Translate
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt", padding=True)
outputs = model.generate(**inputs)
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(translation)  # "Bonjour, comment allez-vous?"
````

# FUTURE IMPROVMENTS

The Helsinki Transformer model used in this project is not perfect — it provides good baseline translations but still has several limitations. Like most open multilingual models, it can produce errors in grammar, tone, or context, especially for low-resource languages or longer sentences.

• Add evalulation in training

• Apply quantization, pruning, and model distillation to reduce size and latency.

• Improve dataset quality with Learning Rate Scheduling, Early Stopping, Gradient Clipping, Data Augmentation.

• Enhanceing Model Architecture 



