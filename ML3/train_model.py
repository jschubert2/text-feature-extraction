import json
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

# Define a color palette with fixed hex codes
color_palette = {
    "green": "#25cc53",
    "yellow": "#d3ff36",
    "orange": "#e39a24",
    "cyan": "#29d0d6",
    "black": "#121212",
    "white": "#f0f0f0",
    "blue": "#3a47d6",
    "red": "#eb1c1c",
    "pink": "#f02ec2",
}
color_classes = list(color_palette.keys())
color_to_label = {color: idx for idx, color in enumerate(color_classes)}

# Load the JSON dataset
with open('trainingprompts_classification.json', 'r') as f:
    dataset = json.load(f)

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Preprocess function for dataset
def preprocess_example(example):
    tokens = tokenizer(example["prompt"], truncation=True, padding="max_length", max_length=64, return_tensors="pt")
    return {
        "input_ids": tokens["input_ids"].squeeze(0),
        "attention_mask": tokens["attention_mask"].squeeze(0),
        "color": torch.tensor(color_to_label[example["color"]], dtype=torch.long),
        "shape": torch.tensor(example["shape"], dtype=torch.float)
    }

# Process the dataset
processed_data = [preprocess_example(entry) for entry in dataset]

# Custom Dataset class
class ColorShapeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Split data and create DataLoaders
train_data, val_data = train_test_split(processed_data, test_size=0.2, random_state=42)
train_loader = DataLoader(ColorShapeDataset(train_data), batch_size=8, shuffle=True)
val_loader = DataLoader(ColorShapeDataset(val_data), batch_size=8, shuffle=False)

# Define the model
class ColorShapeModel(nn.Module):
    def __init__(self):
        super(ColorShapeModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.color_head = nn.Linear(self.bert.config.hidden_size, len(color_palette))  # Color classification
        self.shape_head = nn.Linear(self.bert.config.hidden_size, 4)  # Shape classification

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        color = self.color_head(pooled_output)
        shape_logits = self.shape_head(pooled_output)
        return color, shape_logits

# Initialize model, loss functions, and optimizer
model = ColorShapeModel()
color_loss_fn = nn.CrossEntropyLoss()
shape_loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Training function
def train_model(model, dataloader, optimizer):
    model.train()
    for batch in dataloader:
        input_ids, attention_mask, target_color, target_shape = (
            batch["input_ids"], batch["attention_mask"], batch["color"], batch["shape"]
        )
        optimizer.zero_grad()

        # Forward pass: get predicted color logits and shape logits
        predicted_color, predicted_shape_logits = model(input_ids, attention_mask)

        # For color classification, target_color is an integer class index (not one-hot)
        color_loss = color_loss_fn(predicted_color, target_color)

        # For shape classification, target_shape is one-hot encoded, so we need to use .argmax to get the correct index
        shape_loss = shape_loss_fn(predicted_shape_logits, target_shape.argmax(dim=1))

        # Total loss (sum of color and shape losses)
        loss = color_loss + shape_loss

        # Backpropagation and optimizer step
        loss.backward()
        optimizer.step()

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    color_accuracy, correct_shape_preds = 0, 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, target_color, target_shape = (
                batch["input_ids"], batch["attention_mask"], batch["color"], batch["shape"]
            )
            predicted_color, predicted_shape_logits = model(input_ids, attention_mask)

            # Get predicted color class (logits to class index)
            _, predicted_color_idx = torch.max(predicted_color, dim=1)

            # Compute color accuracy
            correct_color_preds = (predicted_color_idx == target_color).sum().item()
            color_accuracy += correct_color_preds
            total += target_color.size(0)

            # Compute shape accuracy
            _, predicted_shape = torch.max(predicted_shape_logits, 1)
            correct_shape_preds += (predicted_shape == target_shape.argmax(dim=1)).sum().item()

    shape_accuracy = correct_shape_preds / len(dataloader.dataset)
    color_accuracy = color_accuracy / total  # Color accuracy
    print(f"Color Accuracy: {color_accuracy * 100:.2f}%")
    print(f"Shape Accuracy: {shape_accuracy * 100:.2f}%")
# Prediction function
def predict_color_and_shape(prompt):
    tokens = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    model.eval()
    with torch.no_grad():
        predicted_color, predicted_shape_logits = model(input_ids, attention_mask)
    predicted_color_idx = torch.argmax(predicted_color).item()
    predicted_color_hex = color_palette[color_classes[predicted_color_idx]]
    shape_labels = ["circle", "square", "triangle", "star"]
    predicted_shape = shape_labels[torch.argmax(predicted_shape_logits).item()]
    print(f"Prompt: '{prompt}'")
    print(f"Predicted Color (Hex): {predicted_color_hex}")
    print(f"Predicted Shape: {predicted_shape}")

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_model(model, train_loader, optimizer)
    evaluate_model(model, val_loader)

# Sample prompt prediction
sample_prompt = "sunny day"
predict_color_and_shape(sample_prompt)

# Save the model
def save_model(model, path="color_shape_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

save_model(model)
