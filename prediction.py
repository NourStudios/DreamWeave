import os
import json
import imageio.v2 as imageio
import numpy as np

def get_pixel_values(image_path):
    try:
        image = imageio.imread(image_path, pilmode='L')  # Read image in grayscale mode
        pixel_values = np.array(image, dtype=float) * 0.003921568627451  # Scale pixel values to the range [0, 1]
        return pixel_values.flatten()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return []

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def calculate_output(input_values, weights):
    return sigmoid(np.dot(input_values, weights))  # Use dot product for element-wise multiplication and summation

def initialize_weights(num_weights):
    return np.random.uniform(-1, 1, num_weights)

def update_weights(input_values, weights, target_output, learning_rate):
    predicted_output = calculate_output(input_values, weights)
    error = target_output - predicted_output
    weights += learning_rate * error * input_values
    return weights

def predict_object(image_path, model_folder, learning_rate):
    print(f"Predicting object for image: {image_path}")
    try:
        pixel_values = get_pixel_values(image_path)
        object_folders = [f.path for f in os.scandir(model_folder) if f.is_dir()]
        max_prediction = float('-inf')
        best_folder = None

        for object_folder in object_folders:
            weights_file_path = os.path.join(object_folder, "weights.json")
            with open(weights_file_path, 'r') as file:
                weights = np.array(json.load(file))

            prediction = calculate_output(pixel_values, weights)
            if prediction > max_prediction:
                max_prediction = prediction
                best_folder = object_folder

        print(f"Highest predicted output: {max_prediction}")
        if best_folder:
            print(f"Folder with highest prediction: {os.path.basename(best_folder)}")
            user_feedback = input("Is this prediction correct? (y/n): ")
            correct_object = os.path.basename(best_folder)
            if user_feedback.lower() == 'n':
                correct_object = input("Enter the correct object: ")

            # Update weights
            target_outputs = np.zeros(len(object_folders))
            for i, folder in enumerate(object_folders):
                if os.path.basename(folder) == correct_object:
                    target_outputs[i] = 1.0
            weights_file_paths = [os.path.join(folder, "weights.json") for folder in object_folders]
            for weights_file_path in weights_file_paths:
                with open(weights_file_path, 'r') as file:
                    weights = np.array(json.load(file))
                target_output = target_outputs[weights_file_paths.index(weights_file_path)]
                weights = update_weights(pixel_values, weights, target_output, learning_rate)
                with open(weights_file_path, 'w') as file:
                    json.dump(weights.tolist(), file)  # Save updated weights
            print("Weights updated.")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

if __name__ == "__main__":
    image_path = input("Enter the image to predict: ")
    model_folder = input("Enter the model folder containing trained values: ")
    learning_rate = float(1.0)

    predict_object(image_path, model_folder, learning_rate)
