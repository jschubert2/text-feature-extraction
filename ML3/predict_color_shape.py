import torch
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
import json

# Define a color palette with RGB values
color_palette = {
    "green": (37, 204, 83),
    "yellow": (211, 255, 54),
    "orange": (227, 154, 36),
    "cyan": (41, 208, 214),
    "black": (18, 18, 18),
    "white": (240, 240, 240),
    "blue": (58, 71, 214),
    "red": (235, 28, 28),
    "pink": (240, 46, 194),
}
input_json_path = "test_input.json"
output_json_path = "text_prompt_predictions.json"

color_classes = list(color_palette.keys())
color_to_rgb = {idx: rgb for idx, rgb in enumerate(color_palette.values())}

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

    # Convert color logits to RGB
    predicted_color_idx = torch.argmax(predicted_color_logits).item()
    predicted_color_rgb = color_to_rgb[predicted_color_idx]

    # Convert shape logits to label
    shape_labels = ["circle", "square", "triangle", "star"]
    predicted_shape = shape_labels[torch.argmax(predicted_shape_logits).item()]
    one_hot_shape = [1 if i == predicted_shape else 0 for i in range(len(shape_labels))]

    # Assume predicted_shape_logits is already one-hot encoded
    one_hot_shape = predicted_shape_logits.squeeze().tolist()  # Convert tensor to list

    # Return structured output
    return {
        "prompt": prompt,
        "predicted_color_rgb": predicted_color_rgb,
        "predicted_shape": predicted_shape
    }

# Load the JSON file
def process_json(file_path, output_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Results list to store predictions
        results = []

        if "text_prompts" in data:
            for entry in data["text_prompts"]:
                prompt_text = entry.get("text")
                if prompt_text:
                    # Get prediction
                    prediction = predict_color_and_shape(prompt_text)
                    results.append(prediction)
                else:
                    print("Skipped entry with missing 'text'.")

        # Save results to JSON
        with open(output_path, 'w') as outfile:
            json.dump(results, outfile, indent=4)
            print(f"Results saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

process_json(input_json_path, output_json_path)
