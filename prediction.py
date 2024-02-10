import os
import json
import cv2
import numpy as np

LEARNING_RATE = 0.01

def get_pixel_values(image_path):
    """Converts an image to grayscale and returns normalized pixel values."""
    try:
        # Read the image using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Check if the image was successfully loaded
        if image is None:
            raise ValueError(f"Unable to read image {image_path}")

        # Convert pixel values to floating point and normalize
        pixel_values = image.flatten().astype(float) / 255.0

        return pixel_values
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return []

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def calculate_output(input_values, weights):
    """Calculates the output of the neural network."""
    output = np.dot(input_values, weights)
    return sigmoid(output)

def load_weights(model_folder, input_values):
    """Loads weights from model_folder and returns the folder with highest prediction."""
    try:
        object_folders = [f.path for f in os.scandir(model_folder) if f.is_dir()]
        max_prediction = float('-inf')
        best_folder = None

        for object_folder in object_folders:
            weights_file_path = os.path.join(object_folder, "weights.json")
            with open(weights_file_path, 'r') as file:
                weights = json.load(file)

            prediction = calculate_output(input_values, weights)
            if prediction > max_prediction:
                max_prediction = prediction
                best_folder = object_folder

        return best_folder, max_prediction
    except Exception as e:
        print(f"Error loading weights: {e}")
        return None, 0.0

def update_weights(input_values, weights, target_output):
    """Updates weights based on the error between predicted and target output."""
    predicted_output = calculate_output(input_values, weights)
    error = target_output - predicted_output
    for i in range(len(weights)):
        weights[i] += LEARNING_RATE * error * input_values[i]
    return weights

def predict_object(image_path, model_folder):
    """Predicts the object in the image and updates weights based on user feedback."""
    print(f"Predicting object for image: {image_path}")
    try:
        pixel_values = get_pixel_values(image_path)
        best_folder, max_prediction = load_weights(model_folder, pixel_values)
        print(f"Highest predicted output: {max_prediction}")
        if best_folder:
            print(f"Folder with highest prediction: {os.path.basename(best_folder)}")
            user_feedback = input("Is this prediction correct? (y/n): ")
            if user_feedback.lower() == 'n':
                correct_object = input("Enter the correct object name: ")
                weights_file_path = os.path.join(best_folder, "weights.json")
                with open(weights_file_path, 'r') as file:
                    weights = json.load(file)
                updated_weights = update_weights(pixel_values, weights, correct_object)
                with open(weights_file_path, 'w') as file:
                    json.dump(updated_weights, file)
                print("Weights updated based on user feedback.")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

if __name__ == "__main__":
    image_path = input("Enter the image to predict: ")
    model_folder = input("Enter the model folder containing trained values: ")

    predict_object(image_path, model_folder)
