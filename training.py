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

def calculate_output(input_values, weights):
    return sigmoid(np.dot(input_values, weights))  # Use dot product for element-wise multiplication and summation

def initialize_weights(num_weights):
    return [random.uniform(-1, 1) for _ in range(num_weights)]

def update_weights(input_values, weights, target_output):
    predicted_output = calculate_output(input_values, weights)
    error = target_output - predicted_output
    for i in range(len(weights)):
        weights[i] += error * input_values[i]
        weights[i] = round(weights[i], 2)  # Round to two decimal places
    return weights

def save_weights(weights, output_folder, object_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    object_folder = os.path.join(output_folder, object_name)

    if not os.path.exists(object_folder):
        os.makedirs(object_folder)

    output_file_path = os.path.join(object_folder, "weights.json")

    with open(output_file_path, 'w') as output_file:
        json.dump(weights, output_file, indent=2)  # Save weights with indentation for better readability

def train_model(data_folder, output_folder, iterations):
    print(f"Training model using data from folder: {data_folder}")

    learning_rate = 1.0

    for _ in range(iterations):
        total_correct_predictions = 0
        total_predictions = 0

        for folder_name in os.listdir(data_folder):
            folder_path = os.path.join(data_folder, folder_name)
            if not os.path.isdir(folder_path):
                continue

            print(f"Training for object: {folder_name}")

            weights = None

            for filename in os.listdir(folder_path):
                image_path = os.path.join(folder_path, filename)

                try:
                    pixel_values = get_pixel_values(image_path)
                    num_weights = len(pixel_values)
                    if weights is None:
                        weights = initialize_weights(num_weights)
                    predicted_output = calculate_output(pixel_values, weights)
                    target_output = 1.0 if folder_name == filename.split('_')[0] else 0.0
                    if round(predicted_output) == target_output:
                        total_correct_predictions += 1
                    total_predictions += 1
                    update_weights(pixel_values, weights, target_output)
                except Exception as e:
                    print(f"Error processing image {filename}: {e}")

            if weights is not None:
                save_weights(weights, output_folder, folder_name)
                print(f"Model trained and saved for object: {folder_name}")
            else:
                print(f"No images processed for object: {folder_name}")

        accuracy = total_correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"Iteration {_ + 1} - Training completed. Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    data_folder = input("Enter the data folder containing subfolders with images: ")
    model_folder = input("Enter the model folder to save trained values: ")
    iterations = int(input("Enter the number of iterations: "))

    train_model(data_folder, model_folder, iterations)
 