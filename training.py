import os
import json
import imageio.v2 as imageio
import random
import numpy as np

def get_pixel_values(image_path):
    try:
        image = imageio.imread(image_path, mode='L')  # Read image in grayscale mode
        pixel_values = np.array(image, dtype=float) / 255.0
        return pixel_values.flatten()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return []

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def calculate_output(input_values, weights):
    return sigmoid(np.dot(input_values, weights))  # Use dot product for element-wise multiplication and summation

def initialize_weights(num_weights):
    return [random.uniform(-1, 1) for _ in range(num_weights)]

def update_weights(input_values, weights, target_output, learning_rate, object_folder):
    predicted_output = calculate_output(input_values, weights)
    error = target_output - predicted_output
    for i in range(len(weights)):
        weights[i] += learning_rate * error * input_values[i]

    # Remove the filename 'weights.json' from object_folder if it exists
    if object_folder.endswith("weights.json"):
        object_folder = os.path.dirname(object_folder)

    save_weights(weights, object_folder, "weights.json")  # Save updated weights
    return weights

def save_weights(weights, output_folder, object_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    object_folder = os.path.join(output_folder, object_name)

    if not os.path.exists(object_folder):
        os.makedirs(object_folder)

    output_file_path = os.path.join(object_folder, "weights.json")

    with open(output_file_path, 'w') as output_file:
        json.dump(weights, output_file)

def train_model(data_folder, output_folder, iterations):
    print(f"Training model using data from folder: {data_folder}")

    learning_rate = 0.01
    correct_predictions = 0
    total_predictions = 0

    for folder_name in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        weights = None

        for _ in range(iterations):
            for filename in os.listdir(folder_path):
                image_path = os.path.join(folder_path, filename)

                try:
                    pixel_values = get_pixel_values(image_path)
                    num_weights = len(pixel_values)
                    if weights is None:
                        weights = initialize_weights(num_weights)
                    predicted_output = calculate_output(pixel_values, weights)
                    if predicted_output >= 0.5 and folder_name == 'positive':
                        correct_predictions += 1
                    elif predicted_output < 0.5 and folder_name == 'negative':
                        correct_predictions += 1
                    total_predictions += 1
                    # Pass object_folder to update_weights function
                    object_folder = os.path.join(output_folder, folder_name)
                    update_weights(pixel_values, weights, 1.0 if folder_name == 'positive' else 0.0, learning_rate, object_folder)
                except Exception as e:
                    print(f"Error processing image {filename}: {e}")

        if weights is not None:
            save_weights(weights, output_folder, folder_name)
            print(f"Model trained and saved for object: {folder_name}")
        else:
            print(f"No images processed for object: {folder_name}")

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Training accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    data_folder = input("Enter the data folder containing subfolders with images: ")
    model_folder = input("Enter the model folder to save trained values: ")
    iterations = int(input("Enter the number of iterations: "))

    train_model(data_folder, model_folder, iterations)
