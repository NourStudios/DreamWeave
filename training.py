import os
import json
import numpy as np
import imageio.v2 as imageio

def get_pixel_values(image_path):
    try:
        image = imageio.imread(image_path, pilmode='L')  # Read image in grayscale mode
        pixel_values = np.array(image, dtype=float) * 0.003921568627451  # Scale pixel values to the range [0, 1]
        pixel_values = pixel_values.flatten()  # Flatten the array to 1D
        return pixel_values
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return []

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_weights(num_weights):
    return np.random.uniform(-1, 1, num_weights)

def calculate_output(input_values, weights):
    return sigmoid(np.dot(input_values, weights))  # Use dot product for element-wise multiplication and summation

def update_weights(input_values, weights, target_output, learning_rate):
    predicted_output = calculate_output(input_values, weights)
    error = target_output - predicted_output
    weights += learning_rate * error * input_values
    return weights

def save_weights(weights, output_folder, object_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    object_folder = os.path.join(output_folder, object_name)

    if not os.path.exists(object_folder):
        os.makedirs(object_folder)

    output_file_path = os.path.join(object_folder, "weights.json")

    with open(output_file_path, 'w') as output_file:
        json.dump(weights.tolist(), output_file, indent=2)  # Save weights with indentation for better readability

def train_model(data_folder, output_folder, iterations, learning_rate):
    print(f"Training model using data from folder: {data_folder}")

    for _ in range(iterations):
        total_correct_predictions = 0
        total_predictions = 0

        for folder_name in os.listdir(data_folder):
            folder_path = os.path.join(data_folder, folder_name)
            if not os.path.isdir(folder_path):
                continue

            print(f"Training for object: {folder_name}")

            weights = None
            num_weights = None

            for filename in os.listdir(folder_path):
                image_path = os.path.join(folder_path, filename)

                try:
                    pixel_values = get_pixel_values(image_path)
                    if num_weights is None:
                        num_weights = len(pixel_values)
                        weights = initialize_weights(num_weights)
                    predicted_output = calculate_output(pixel_values, weights)
                    target_output = 1.0 if folder_name == filename.split('_')[0] else 0.0
                    if round(predicted_output) == target_output:
                        total_correct_predictions += 1
                    total_predictions += 1
                    weights = update_weights(pixel_values, weights, target_output, learning_rate)
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
    learning_rate = float(1.0)

    train_model(data_folder, model_folder, iterations, learning_rate)
