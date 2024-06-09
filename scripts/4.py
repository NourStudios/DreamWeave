import os
import json
import numpy as np
from PIL import Image

def get_brightness_values(image_path):
    try:
        image = Image.open(image_path)
        image = image.convert('L')
        image = image.resize((28, 28))
        brightness_values = np.array(image)
        brightness_values_flat = brightness_values.flatten()
        brightness_values_flat = brightness_values_flat / 255.0
        return brightness_values_flat
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=0)

def load_weights(output_folder, layer_name):
    layer_folder = os.path.join(output_folder, layer_name)
    output_file_path = os.path.join(layer_folder, "weights.json")
    with open(output_file_path, 'r') as input_file:
        data = json.load(input_file)
        weights = np.array(data['weights'])
        bias = np.array(data['bias'])
    return weights, bias

def calculate_output(input_values, weights, bias):
    return softmax(np.dot(input_values, weights) + bias)

def predict(image_path, model_folder):
    input_values = get_brightness_values(image_path)
    if input_values is None:
        return None, None

    layer = 1
    while True:
        layer_name = f"layer{layer}"
        layer_folder = os.path.join(model_folder, layer_name)
        if not os.path.exists(layer_folder) or not os.path.isfile(os.path.join(layer_folder, "weights.json")):
            break
        weights, bias = load_weights(model_folder, layer_name)
        input_values = calculate_output(input_values, weights, bias)
        layer += 1

    return np.argmax(input_values), input_values

if __name__ == "__main__":
    image_path = input("Enter the path to the image file: ")
    model_folder = input("Enter the model folder containing the trained values: ")

    predicted_class, output_probabilities = predict(image_path, model_folder)
    if predicted_class is not None:
        print(f"Predicted Class: {predicted_class}")
        print(f"Output Probabilities: {output_probabilities}")
    else:
        print("Prediction failed.")
        