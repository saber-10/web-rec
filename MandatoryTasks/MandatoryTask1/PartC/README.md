
# TASK 1.C: QUESTION ANSWERING (NEWSQA DATASET)

This task experiments with building and fine-tuning a Question Answering (QA) model using the NewsQA dataset.

A transformer-based architecture  is fine-tuned to extract precise answers from news article passages.

## 🧠 Transformer for Question Answering

The model fine-tuned in this task is based on Hugging Face Transformer, a pre-trained RoBERTa-base (or BERT-base) architecture. RoBERTa stands for Robustly Optimized BERT Approach.


It is fine-tuned on the NewsQA dataset — a large collection of news articles and associated comprehension-style question.

## 🚀 PROJECT OVERVIEW

The notebook builds a complete extractive QA pipeline using the Hugging Face ecosystem.

It performs:

1. Dataset loading from Hugging Face Datasets

2. Data preprocessing and context-question alignment

3. Tokenization and feature creation with overflow handling

4. Fine-tuning of a transformer model using Trainer

6. Inference for custom question-answering

## 📦 DEPENDENCIES

The following libraries were used:

• transformers — for model, tokenizer, and training utilities

• datasets — to load and manage the NewsQA dataset

• torch — PyTorch backend for training

• numpy & pandas — for data manipulation

• tqdm — for progress visualization

Install all dependencies:

````markdown
pip install transformers datasets torch evaluate pandas numpy tqdm
````
 ## 🪜 PIPELINE STEPS

### 1. LOADING THE DATASET

Source: lucadiliello/newsqa from Hugging Face Datasets

Domain: News articles and comprehension-style questions

#### Structure:

1. context: context for the given question

2. question: question

3. answer: answer for the question

4. key: unique value to identify example

5. label: dict containing start and end of answer

````markdown
from datasets import load_dataset
ds = load_dataset("lucadiliello/newsqa")
````

### 2. 🧹 DATA PREPARATION & PREPROCESSING

This is the most crucial stage of the QA pipeline. It ensures that the question–context pairs are correctly aligned and that each tokenized sample is properly labeled for training the model to extract the correct answer span.

#### 🔧 Key Steps:

1. Question Cleaning — Removes leading/trailing spaces from the question text.

2. Tokenization with Sliding Window —

• Uses the tokenizer to split text into tokens with overlap (stride=128).

• Ensures that long contexts exceeding max_length=384 are chunked properly.

3. Overflow Mapping — Keeps track of which tokenized chunks belong to which original sample (overflow_to_sample_mapping).

4. Offset Mapping — For each sub-token returned by the tokenizer, the offset mapping gives us a tuple indicating the sub-token’s start position and end position relative to the original token it was split from.

5. Answer Alignment —

• For each tokenized chunk, locates where the answer text lies.

• Converts those into token indices (start_positions and end_positions) for model training.

6. Output — Returns a dictionary containing:

• Tokenized input (input_ids, attention_mask, etc.)

• Corresponding start_positions and end_positions labels.

### 3. MODEL FINE-TUNING

#### Base Model:

• Architecture: RoBERTa / BERT (encoder-only transformer)

• Task: Extractive Question Answering

• Pre-trained model: roberta-base

• Parameters: ~125M

#### Training Parameters 

````markdown
from transformers import TrainingArguments

args = TrainingArguments(
    "roberta-base-finetuned",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=1,
    weight_decay=0.01,
    fp16=True,
    gradient_accumulation_steps=2,
)
````

### 5. MAKING PREDICTIONS

````markdown
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

context = "amey have music club audition tommorow and he's gona get fucked real bad"
question = ["what does amey have tommorow?"]

result = qa_pipeline(
    question=question,
    context=context
)
print(result["answer"])
#output - music club audition
````

## 🌟 FUTURE IMPROVEMENTS

• Use larger architectures (RoBERTa-large, DeBERTa-v3) for improved comprehension

• Integrate early stopping and learning rate scheduling

• Apply mixed-precision training (fp16) for faster computation

## 🧑‍💻 AUTHOR

Amey Bhagat

📧 amey.241ds009@gmail.com


