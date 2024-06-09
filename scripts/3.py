import os
import json
import numpy as np

def get_brightness_values(text_file_path):
    try:
        with open(text_file_path, 'r') as file:
            brightness_values = np.array([float(x) for x in file.read().split()])
        if brightness_values.size != 784:  # Ensure it matches the expected size (28x28)
            raise ValueError("Unexpected number of brightness values in the text file.")
        brightness_values = brightness_values / 255.0  # Normalize pixel values
        return brightness_values
    except Exception as e:
        print(f"Error processing text file {text_file_path}: {e}")
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

def predict(text_file_path, model_folder):
    input_values = get_brightness_values(text_file_path)
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
    text_file_path = input("Enter the path to the text file: ")
    model_folder = input("Enter the model folder containing the trained values: ")

    predicted_class, output_probabilities = predict(text_file_path, model_folder)
    if predicted_class is not None:
        print(f"Predicted Class: {predicted_class}")
        print(f"Output Probabilities: {output_probabilities}")
    else:
        print("Prediction failed.")
