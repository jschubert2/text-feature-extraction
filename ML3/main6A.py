import torch
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np

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
color_to_hex = {idx: hex_code for idx, hex_code in enumerate(color_palette.values())}

# Define the model architecture for color classification
class ColorShapeModel(torch.nn.Module):
    def __init__(self):
        super(ColorShapeModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.color_head = torch.nn.Linear(self.bert.config.hidden_size, len(color_palette))  # Color classification
        self.shape_head = torch.nn.Linear(self.bert.config.hidden_size, 4)  # Shape classification

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        color_logits = self.color_head(pooled_output)  # Color classification output
        shape_logits = self.shape_head(pooled_output)  # Shape classification output
        return color_logits, shape_logits

# Load the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = ColorShapeModel()

# Load the model parameters
model.load_state_dict(torch.load("color_shape_model.pth"))
model.eval()  # Set the model to evaluation mode

# Function to generate predictions for a custom prompt
def predict_color_and_shape(prompt):
    # Tokenize the input prompt
    tokens = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    # Make prediction
    with torch.no_grad():
        predicted_color_logits, predicted_shape_logits = model(input_ids, attention_mask)

    # Convert color logits to hex
    predicted_color_idx = torch.argmax(predicted_color_logits).item()
    predicted_color_hex = color_to_hex[predicted_color_idx]

    # Convert shape logits to label
    shape_labels = ["circle", "square", "triangle", "star"]
    predicted_shape = shape_labels[torch.argmax(predicted_shape_logits).item()]

    print(f"Prompt: '{prompt}'")
    print(f"Predicted Color (Hex): {predicted_color_hex}")
    print(f"Predicted Shape: {predicted_shape}")

# Run the prediction for a custom prompt
#custom_prompt = "a mysterious, foggy forest at dawn"
custom_prompt = "a sunny afternoon with friends"
predict_color_and_shape(custom_prompt)
