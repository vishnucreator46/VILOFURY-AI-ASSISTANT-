# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 21:06:19 2025

@author: vishn
"""

import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Load preprocessed embeddings and labels
with open("processed_vilofury.pkl", "rb") as f:
    embeddings, labels, le = pickle.load(f)

# Convert embeddings and labels to PyTorch tensors
X = torch.tensor(embeddings, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.long)

print(f"🔹 Data ready for training: X shape {X.shape}, y shape {y.shape}")



class IntentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(IntentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

input_dim = X.shape[1]       # 768 from DistilBERT
hidden_dim = 256
output_dim = len(set(labels))  # number of intents

model = IntentClassifier(input_dim, hidden_dim, output_dim)



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


epochs = 20  # you can increase for better accuracy

for epoch in range(epochs):
    running_loss = 0.0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs} — Loss: {running_loss/len(dataloader):.4f}")

print("✅ Training complete!")


# Save only model weights
torch.save(model.state_dict(), "vilofury_model_weights.pth")

# Save label encoder separately
with open("vilofury_label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Model weights and label encoder saved separately!")

