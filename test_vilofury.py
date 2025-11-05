import pickle
import json
import torch
import pickle
from transformers import DistilBertTokenizer, DistilBertModel
import random

import torch
from transformers import DistilBertTokenizer, DistilBertModel

# Load tokenizer and pretrained DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Load label encoder
with open("vilofury_label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Rebuild the classifier
class IntentClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(IntentClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

input_dim = 768
hidden_dim = 256
output_dim = len(le.classes_)
model = IntentClassifier(input_dim, hidden_dim, output_dim)

# Load model weights
model.load_state_dict(torch.load("vilofury_model_weights.pth", map_location=torch.device('cpu')))
model.eval()


with open("intents.json", "r") as f:
    intents = json.load(f)
    
    
    
    
def get_response(message):
    # Tokenize user message
    encodings = tokenizer(message, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = bert_model(**encodings)
        embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
    
    # Predict intent
    logits = model(embedding)
    predicted_idx = torch.argmax(logits, dim=1).item()
    tag = le.inverse_transform([predicted_idx])[0]
    
    # Select random response
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])
            return f"[{tag}] → {response}"
        



print("🤖 ViloFury is ready! Type 'quit' to exit.")

while True:
    message = input("You: ")
    if message.lower() == "quit":
        print("ViloFury: Bye! 👋")
        break
    response = get_response(message)
    print("ViloFury:", response)

