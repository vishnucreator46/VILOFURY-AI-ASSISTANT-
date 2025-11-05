from transformers import DistilBertTokenizer, DistilBertModel
import torch

print("🔄 Loading DistilBERT model (PyTorch)...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
print("✅ DistilBERT loaded successfully!")


import json

# Load intents file
with open("intents.json", "r") as file:
    data = json.load(file)

sentences = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])

print(f"🔹 Loaded {len(sentences)} patterns from intents.json")


import torch

# Tokenize the sentences
encodings = tokenizer(
    sentences,
    padding=True,      # pad shorter sequences
    truncation=True,   # truncate long sequences
    return_tensors="pt"
)

print("🔹 Tokenization complete. Keys:", encodings.keys())

# Generate embeddings (sentence representations)
with torch.no_grad():
    outputs = model(**encodings)
    embeddings = outputs.last_hidden_state[:, 0, :]  # Use the [CLS] token

print(f"🔹 Embeddings generated. Shape: {embeddings.shape}")

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(labels)

print(f"🔹 Labels encoded. Classes: {list(le.classes_)}")


import pickle

with open("processed_vilofury.pkl", "wb") as f:
    pickle.dump((embeddings, y, le), f)

print("✅ Preprocessed data saved as processed_vilofury.pkl")


