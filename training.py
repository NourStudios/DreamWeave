import os
import json
import uuid
from PIL import Image

def get_pixel_values(image_path):
    image = Image.open(image_path)
    return [int(pixel) for pixel in image.convert('1').tobytes()]

def calculate_values(previous_values):
    max_value = max(previous_values)
    normalized_values = [value / max_value for value in previous_values]
    combined_values = [normalized_values[i] + normalized_values[i + 1] for i in range(len(normalized_values) - 1)]
    return combined_values

def save_values(values, output_folder, object_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    object_folder = os.path.join(output_folder, object_name)

    if not os.path.exists(object_folder):
        os.makedirs(object_folder)

    random_filename = f"{uuid.uuid4().hex}_result.json"
    output_file_path = os.path.join(object_folder, random_filename)

    with open(output_file_path, 'w') as output_file:
        json.dump(values, output_file)

def train_model(image_folder, output_folder):
    print(f"Training model for {image_folder}...")

    combined_values_list = []

    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            pixel_values = get_pixel_values(image_path)
            combined_values = calculate_values(pixel_values)
            combined_values_list.append(combined_values)

    if combined_values_list:
        # Calculate average values
        averaged_values = [sum(x) / len(x) for x in zip(*combined_values_list)]
        save_values(averaged_values, output_folder, os.path.basename(image_folder))
        print(f"Model trained and saved for {image_folder}")

if __name__ == "__main__":
    image_folder = input("Enter the folder containing training images: ")
    model_folder = input("Enter the model folder to save trained values: ")

    train_model(image_folder, model_folder)
