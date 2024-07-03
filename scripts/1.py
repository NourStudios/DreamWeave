import os
import json
import numpy as np
import random
from PIL import Image

def text_to_values(text_path, max_length=500):
    try:
        with open(text_path, 'r') as file:
            text = file.read()
    except Exception as e:
        print(f"Error reading file {text_path}: {e}")
        return None
    
    char_values = [ord(char) for char in text[:max_length]]
    char_values.extend([0] * (max_length - len(char_values)))

    # Normalize and scale values to range [0.001, 1]
    normalized_values = [value / 127 for value in char_values]
    scaled_values = [0.001 + (value * (1 - 0.001)) for value in normalized_values]
    
    return np.array(scaled_values)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=0)

def initialize_weights(num_features, num_classes):
    stddev = np.sqrt(2.0 / num_features)
    weights = np.random.normal(0, stddev, (num_features, num_classes))
    bias = np.zeros(num_classes)
    return weights, bias

def calculate_output(input_values, weights, bias):
    return softmax(np.dot(input_values, weights) + bias)

def mean_squared_error(predicted_output, target_output):
    return 0.5 * np.sum((predicted_output - target_output) ** 2)

def update_weights(input_values, weights, bias, target_output, predicted_output, learning_rate):
    error = predicted_output - target_output
    dL_dw = np.outer(input_values, error)
    dL_db = error
    weights -= learning_rate * dL_dw
    bias -= learning_rate * dL_db
    return weights, bias

def save_weights(weights, bias, output_folder, layer_name, object_names):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    layer_folder = os.path.join(output_folder, layer_name)
    if not os.path.exists(layer_folder):
        os.makedirs(layer_folder)
    output_file_path = os.path.join(layer_folder, "weights.json")
    data = {'weights': weights.tolist(), 'bias': bias.tolist(), 'object_names': object_names}
    with open(output_file_path, 'w') as output_file:
        json.dump(data, output_file, indent=2)

def train_model(data_folder, output_folder, iterations, learning_rate):
    print(f"Training model using data from folder: {data_folder}")

    layer = 1
    while True:
        layer_folder = os.path.join(data_folder, f"layer{layer}")
        if not os.path.exists(layer_folder):
            break
        print(f"Processing {layer_folder}")

        class_to_index = {folder_name: i for i, folder_name in enumerate(os.listdir(layer_folder))}
        num_classes = len(class_to_index)

        weights, bias = None, None
        num_features = None
        all_data = []

        for folder_name in os.listdir(layer_folder):
            folder_path = os.path.join(layer_folder, folder_name)
            if not os.path.isdir(folder_path):
                continue

            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)

                features = text_to_values(file_path)
                if features is None:
                    print(f"Skipping file {filename} due to processing error")
                    continue

                if num_features is None:
                    num_features = len(features)
                    weights, bias = initialize_weights(num_features, num_classes)

                class_index = class_to_index[folder_name]
                target_output = np.full(num_classes, 0.001)
                target_output[class_index] = 1.0

                all_data.append((features, target_output))

        for iteration in range(iterations):
            total_correct_predictions = 0
            total_predictions = 0
            total_loss = 0

            random.shuffle(all_data)

            for features, target_output in all_data:
                predicted_output = calculate_output(features, weights, bias)
                loss = mean_squared_error(predicted_output, target_output)
                total_loss += loss

                if np.argmax(predicted_output) == np.argmax(target_output):
                    total_correct_predictions += 1
                total_predictions += 1

                weights, bias = update_weights(features, weights, bias, target_output, predicted_output, learning_rate)

            accuracy = total_correct_predictions / total_predictions if total_predictions > 0 else 0
            average_loss = total_loss / total_predictions if total_predictions > 0 else 0
            print(f"Layer {layer}, Iteration {iteration + 1} - Accuracy: {accuracy * 100:.2f}%, Average Loss: {average_loss:.4f}")

        save_weights(weights, bias, output_folder, f"layer{layer}", list(class_to_index.keys()))
        print(f"Model for layer {layer} trained and saved.")

        layer += 1

    print("Training completed for all layers.")

if __name__ == "__main__":
    data_folder = input("Enter the data folder containing subfolders with text files: ")
    model_folder = input("Enter the model folder to save trained values: ")
    iterations = int(input("Enter the number of iterations: "))
    learning_rate = float(input("Enter the learning rate: "))

    train_model(data_folder, model_folder, iterations, learning_rate)
