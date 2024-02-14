import os
import json
import imageio.v2 as imageio
import numpy as np

def get_pixel_values(image_path):
    try:
        image = imageio.imread(image_path, mode='L')  # Read image in grayscale mode
        pixel_values = np.array(image, dtype=float) / 255.0
        rounded_values = np.round(pixel_values, decimals=2)  # Round to two decimal places
        return rounded_values.flatten()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return []

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_output(input_values, weights):
    return sigmoid(np.dot(input_values, weights))  # Use dot product for element-wise multiplication and summation

def load_weights(model_folder, input_values):
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

def update_weights(input_values, weights, target_output, object_folder):
    predicted_output = calculate_output(input_values, weights)
    error = target_output - predicted_output
    for i in range(len(weights)):
        weights[i] += error * input_values[i]
        weights[i] = round(weights[i], 2)  # Round to two decimal places
    save_weights(weights, object_folder, "")  # Save updated weights
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

def predict_object(image_path, model_folder):
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
                # Pass object_folder to update_weights function
                object_folder = os.path.join(model_folder, correct_object)
                update_weights(pixel_values, weights, 1.0 if correct_object == os.path.basename(best_folder) else 0.0, object_folder)
                print("Weights updated based on user feedback.")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

if __name__ == "__main__":
    image_path = input("Enter the image to predict: ")
    model_folder = input("Enter the model folder containing trained values: ")

    predict_object(image_path, model_folder)
