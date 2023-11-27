import os
import json
import random
import string
from PIL import Image

def get_pixel_values(image_path):
    image = Image.open(image_path)
    return [int(pixel) for pixel in image.convert('1').tobytes()]

def calculate_values(previous_values):
    max_value = max(previous_values)
    normalized_values = [value / max_value for value in previous_values]
    combined_values = [normalized_values[i] + normalized_values[i + 1] for i in range(len(normalized_values) - 1)]
    return combined_values

def read_values(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def calculate_similarity(predicted_values, trained_values):
    differences = [abs(p - t) for p, t in zip(predicted_values, trained_values)]
    similarity_score = 1 - (sum(differences) / len(differences))
    return similarity_score * 100

def generate_random_filename():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=16))

def save_results(results, learned_folder, object_name):
    object_folder = os.path.join(learned_folder, object_name)

    if not os.path.exists(object_folder):
        os.makedirs(object_folder)

    random_filename = generate_random_filename()
    output_file_path = os.path.join(object_folder, f"{random_filename}_result.json")

    with open(output_file_path, 'w') as output_file:
        json.dump(results, output_file)

def predict_combined_value(image_path, learned_folder):
    print(f"Processing image: {image_path}")

    try:
        image = Image.open(image_path)
        predicted_values = get_pixel_values(image_path)
        predicted_combined_values = calculate_values(predicted_values)

    except Exception as e:
        print(f"Error processing image: {e}")
        return

    highest_similarity = 0
    best_trained_folder = ""

    for folder_name in os.listdir(learned_folder):
        folder_path = os.path.join(learned_folder, folder_name)

        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith('_result.json'):
                    trained_file_path = os.path.join(folder_path, filename)

                    try:
                        trained_values = read_values(trained_file_path)
                    except Exception as e:
                        print(f"Error reading values from {trained_file_path}: {e}")
                        continue

                    similarity = calculate_similarity(predicted_combined_values, trained_values)

                    print(f"{filename}: {similarity:.2f}%")

                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_trained_folder = folder_name

    print(f"\nHighest Similarity: {highest_similarity:.2f}%")

    if best_trained_folder:
        print(f"Recognized object: {best_trained_folder}")
        user_feedback = input("Is this correct? (y/n): ")

        if user_feedback.lower() == 'y':
            save_results(predicted_combined_values, learned_folder, best_trained_folder)
            print("Results saved.")
        else:
            object_name = input("Enter the correct object name: ")
            save_results(predicted_combined_values, learned_folder, object_name)
            print("Results saved.")
    else:
        print("No recognized object.")

if __name__ == "__main__":
    image_path = input("Enter the image to predict: ")
    learned_folder = input("Enter the model folder: ")

    predict_combined_value(image_path, learned_folder)
