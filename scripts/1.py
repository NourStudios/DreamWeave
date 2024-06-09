import os
import json
import numpy as np
import random

def get_features_from_txt(file_path):
    try:
        with open(file_path, 'r') as file:
            data = file.read().strip().split()
            features = np.array(data[:-1], dtype=float)
            target_index = int(data[-1])
            return features, target_index
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None

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

def cross_entropy_loss(predicted_output, target_output):
    return -np.sum(target_output * np.log(predicted_output + 1e-15))

def update_weights(input_values, weights, bias, target_output, predicted_output, learning_rate):
    error = target_output - predicted_output
    dL_dw = -2 * np.outer(input_values, error)
    dL_db = -2 * error
    weights -= learning_rate * dL_dw
    bias -= learning_rate * dL_db
    return weights, bias

def save_weights(weights, bias, output_folder, layer_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    layer_folder = os.path.join(output_folder, layer_name)
    if not os.path.exists(layer_folder):
        os.makedirs(layer_folder)
    output_file_path = os.path.join(layer_folder, "weights.json")
    data = {'weights': weights.tolist(), 'bias': bias.tolist()}
    with open(output_file_path, 'w') as output_file:
        json.dump(data, output_file, indent=2)

def load_weights(output_folder, layer_name):
    layer_folder = os.path.join(output_folder, layer_name)
    output_file_path = os.path.join(layer_folder, "weights.json")
    with open(output_file_path, 'r') as input_file:
        data = json.load(input_file)
        weights = np.array(data['weights'])
        bias = np.array(data['bias'])
    return weights, bias

def train_model(data_folder, output_folder, iterations, learning_rate):
    print(f"Training model using data from folder: {data_folder}")

    num_layers = len([name for name in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, name))])

    for layer in range(1, num_layers + 1):
        layer_folder = os.path.join(data_folder, f"layer{layer}")
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

                features, target_index = get_features_from_txt(file_path)
                if features is None:
                    print(f"Skipping file {filename} due to processing error")
                    continue

                if num_features is None:
                    num_features = len(features)
                    weights, bias = initialize_weights(num_features, num_classes)

                class_index = class_to_index[folder_name]
                target_output = np.zeros(num_classes)
                target_output[class_index] = 1.0

                all_data.append((features, target_output))

        for iteration in range(iterations):
            total_correct_predictions = 0
            total_predictions = 0
            total_loss = 0

            random.shuffle(all_data)

            for features, target_output in all_data:
                predicted_output = calculate_output(features, weights, bias)
                loss = cross_entropy_loss(predicted_output, target_output)
                total_loss += loss

                if np.argmax(predicted_output) == np.argmax(target_output):
                    total_correct_predictions += 1
                total_predictions += 1

                weights, bias = update_weights(features, weights, bias, target_output, predicted_output, learning_rate)

            accuracy = total_correct_predictions / total_predictions if total_predictions > 0 else 0
            average_loss = total_loss / total_predictions if total_predictions > 0 else 0
            print(f"Layer {layer}, Iteration {iteration + 1} - Accuracy: {accuracy * 100:.2f}%, Average Loss: {average_loss:.4f}")

        save_weights(weights, bias, output_folder, f"layer{layer}")
        print(f"Model for layer {layer} trained and saved.")

    print("Training completed for all layers.")

if __name__ == "__main__":
    data_folder = input("Enter the data folder containing subfolders with text files: ")
    model_folder = input("Enter the model folder to save trained values: ")
    iterations = int(input("Enter the number of iterations: "))
    learning_rate = float(input("Enter the learning rate: "))

    train_model(data_folder, model_folder, iterations, learning_rate)
