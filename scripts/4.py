import os
import json
import numpy as np
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

def load_weights(output_folder, layer_name):
    layer_folder = os.path.join(output_folder, layer_name)
    output_file_path = os.path.join(layer_folder, "weights.json")
    with open(output_file_path, 'r') as input_file:
        data = json.load(input_file)
        weights = np.array(data['weights'])
        bias = np.array(data['bias'])
        object_names = data['object_names']
    return weights, bias, object_names

def predict(image_path, model_folder):
    input_values = get_brightness_values(image_path)
    if input_values is None:
        return "Prediction failed due to image processing error."

    layer = 1
    while True:
        try:
            weights, bias, object_names = load_weights(model_folder, f"layer{layer}")
            predicted_output = softmax(np.dot(input_values, weights) + bias)
            predicted_class_index = np.argmax(predicted_output)
            predicted_class = object_names[predicted_class_index]
            print(f"Layer {layer}: Predicted class - {predicted_class}")

            # Check if there is a next layer
            if not os.path.exists(os.path.join(model_folder, f"layer{layer+1}")):
                break

            input_values = np.zeros(len(object_names))
            input_values[predicted_class_index] = 1
            layer += 1
        except FileNotFoundError:
            break

    return predicted_class

if __name__ == "__main__":
    model_folder = input("Enter the model folder: ")
    image_path = input("Enter the path to the image: ")

    predicted_class = predict(image_path, model_folder)
    print(f"Final Prediction: {predicted_class}")
