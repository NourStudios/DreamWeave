import os
import json
import numpy as np

def text_to_values_from_string(text, max_length=500):
    char_values = [ord(char) for char in text[:max_length]]
    char_values.extend([0] * (max_length - len(char_values)))

    # Normalize and scale values to range [0.001, 1]
    normalized_values = [value / 127 for value in char_values]
    scaled_values = [0.001 + (value * (1 - 0.001)) for value in normalized_values]
    
    return np.array(scaled_values)

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

def predict_text(text, model_folder):
    input_values = text_to_values_from_string(text)
    if input_values is None:
        return "Prediction failed due to text processing error."

    layer = 1
    while True:
        try:
            weights, bias, object_names = load_weights(model_folder, f"layer{layer}")
            predicted_output = softmax(np.dot(input_values, weights) + bias)
            predicted_class_index = np.argmax(predicted_output)
            predicted_class = object_names[predicted_class_index]

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
    text = input("Enter the text for prediction: ")

    predicted_class = predict_text(text, model_folder)
    print(f"Message: {predicted_class}")
