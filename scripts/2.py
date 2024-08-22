import os
import json
import numpy as np
import random
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

def initialize_weights(num_features, num_classes):
    stddev = np.sqrt(2.0 / num_features)
    weights = np.random.normal(0, stddev, (num_features, num_classes))
    bias = np.full(num_classes, 0.01)
    return weights, bias

def calculate_output(input_values, weights, bias):
    return softmax(np.dot(input_values, weights) + bias)

def mean_squared_error(predicted_output, target_output):
    return 0.5 * np.sum((predicted_output - target_output) ** 2)

def custom_update_weights(weights, input_values, error, learning_rate):
    error = error.reshape(-1, 1)  # Reshape for multiplication
    input_values = input_values.reshape(-1, 1)  # Reshape for multiplication
    
    # Update the weights
    update = input_values @ error.T  # shape (num_features, num_classes)
    return weights + learning_rate * update

def update_weights(weights, bias, input_values, target_output, predicted_output, learning_rate):
    error = predicted_output - target_output
    dL_dw = np.outer(input_values, error)
    dL_db = error
    
    # Update weights and bias
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

def train_model(data_folder, output_folder, iterations, learning_rate, custom_iterations):
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

                features = get_brightness_values(file_path)
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

        # First phase: using custom weight update method
        for iteration in range(custom_iterations):
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

                output_error = target_output - predicted_output
                weights = custom_update_weights(weights, features, output_error, learning_rate)

            accuracy = total_correct_predictions / total_predictions if total_predictions > 0 else 0
            average_loss = total_loss / total_predictions if total_predictions > 0 else 0

        # Second phase: using gradient descent from scratch
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

                output_error = target_output - predicted_output
                weights, bias = update_weights(weights, bias, features, target_output, predicted_output, learning_rate)

            accuracy = total_correct_predictions / total_predictions if total_predictions > 0 else 0
            average_loss = total_loss / total_predictions if total_predictions > 0 else 0
            print(f"Iteration {iteration + 1} - Accuracy: {accuracy * 100:.2f}%, Average Loss: {average_loss:.4f}")

        save_weights(weights, bias, output_folder, f"layer{layer}", list(class_to_index.keys()))
        print(f"Model for layer {layer} trained and saved.")

        layer += 1

    print("Training completed for all layers.")

if __name__ == "__main__":
    data_folder = "data"
    model_folder = "test2"
    iterations = 0
    custom_iterations = 100  # Number of iterations using the custom update method
    learning_rate = 0.01

    train_model(data_folder, model_folder, iterations, learning_rate, custom_iterations)
