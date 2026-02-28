# Emotion-Detection
The overall system architecture consists of a frontend interface built using Stream lit for user interaction and a backend classification engine powered by a fine-tuned Distil BERT model.

## Features

- User Input: The user enters free-text input expressing thoughts or feelings on the Streamlit interface.
- Preprocessing: The text is sent to the backend. where cleaning, tokenization, and padding occur.
- Embedding: DistilBERT processes the encoded input and generates contextual embeddings.
- Classification: A custom classification head predicts the independent probability of each emotion label.
- Visualization: The frontend displays the detected emotions using color-coded bars, confidence scores, and emojis.

##⚙️Tech Stack
Frontend: Streamlit
Backend: Python, BERT
Database: MySQL

## Installation

numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.2

# PyTorch (CPU build). For GPU, install from pytorch.org per your CUDA version.
torch

# NLP stack
transformers==4.43.3
tokenizers
sentencepiece==0.2.0
accelerate==0.33.0

# App and utils
streamlit
tqdm==4.66.5
Pillow==10.4.0
protobuf==5.27.2


## Outputs

<img width="1920" height="1051" alt="image" src="https://github.com/user-attachments/assets/c867b6f5-f503-4222-a39a-3c4bfb5c6ed8" />


<img width="1920" height="1051" alt="Screenshot From 2025-10-28 14-53-12" src="https://github.com/user-attachments/assets/4219e340-a2da-479a-8d6b-d412e1611129" />


