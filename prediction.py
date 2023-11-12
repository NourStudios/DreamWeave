import os
from PIL import Image
import re

def get_combined_value(pixel):
    if len(pixel) == 3 or len(pixel) == 4:  # RGB or RGBA image
        brightness = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
        return brightness + sum(pixel[:3])  # Consider only the RGB channels
    elif len(pixel) == 1:  # Grayscale image
        return pixel[0]
    elif len(pixel) == 2:  # Grayscale image with alpha channel
        return pixel[0]
    else:
        raise ValueError(f"Unsupported image mode ({len(pixel)} channels). Only RGB, RGBA, and grayscale images are supported.")

def predict_combined_value(model_folder):
    combined_values = []

    for trained_folder_name in os.listdir(model_folder):
        trained_folder_path = os.path.join(model_folder, trained_folder_name)

        if os.path.isdir(trained_folder_path):
            for filename in os.listdir(trained_folder_path):
                if filename.endswith('_results.txt'):
                    result_path = os.path.join(trained_folder_path, filename)

                    with open(result_path, "r") as result_file:
                        for line in result_file:
                            # Extract combined value from the line using regular expressions
                            value = float(re.findall(r'[-+]?\d*\.\d+|\d+', line)[-1])
                            combined_values.append(value)

    return combined_values

def calculate_percentage(predicted_combined_values, trained_folder):
    percentages = []

    for filename in os.listdir(trained_folder):
        if filename.endswith('_results.txt'):
            result_path = os.path.join(trained_folder, filename)

            with open(result_path, "r") as result_file:
                matched_pixels = 0
                total_pixels = 0

                for i, line in enumerate(result_file):
                    if i < len(predicted_combined_values):  # Check the index before accessing the list
                        # Extract combined value from the line using regular expressions
                        value = float(re.findall(r'[-+]?\d*\.\d+|\d+', line)[-1])
                        predicted_combined_value = predicted_combined_values[i]

                        # Compare with predicted values
                        if value == predicted_combined_value:
                            matched_pixels += 1

                        total_pixels += 1

                percentage = (matched_pixels / total_pixels) * 100
                percentages.append((filename, percentage))

    return percentages

def compare_results(predicted_combined_values, model_folder):
    highest_percentage = 0
    best_trained_folder = ""

    for trained_folder_name in os.listdir(model_folder):
        trained_folder_path = os.path.join(model_folder, trained_folder_name)

        if os.path.isdir(trained_folder_path):
            percentages = calculate_percentage(predicted_combined_values, trained_folder_path)

            for filename, percentage in percentages:
                if percentage > highest_percentage:
                    highest_percentage = percentage
                    best_trained_folder = trained_folder_name

                print(f"{filename}: {percentage:.2f}%")

    return highest_percentage, best_trained_folder

def update_feedback(model_folder, best_trained_folder, predicted_combined_values):
    feedback = input("\nIs the recognized object correct? (Y/N): ").lower()

    if feedback == 'y':
        print(f"Object {best_trained_folder} recognized with the highest confidence.")
        result_path = os.path.join(model_folder, f"{best_trained_folder}_results.txt")
        with open(result_path, "w") as result_file:
            for i, combined_value in enumerate(predicted_combined_values):
                result_file.write(f"Pixel {i + 1}: Combined Value = {combined_value:.2f}\n")
    elif feedback == 'n':
        new_folder_name = input("Enter a new name for the recognized object: ")
        new_folder_path = os.path.join(model_folder, new_folder_name)

        os.makedirs(new_folder_path, exist_ok=True)

        # Save the new results with the updated name
        result_path = os.path.join(new_folder_path, f"{new_folder_name}_results.txt")
        with open(result_path, "w") as result_file:
            for i, combined_value in enumerate(predicted_combined_values):
                result_file.write(f"Pixel {i + 1}: Combined Value = {combined_value:.2f}\n")
    else:
        print("Invalid feedback. Please enter 'Y' or 'N'.")

if __name__ == "__main__":
    model_folder = input("Enter the model folder: ")

    predicted_combined_values = predict_combined_value(model_folder)
    highest_percentage, best_trained_folder = compare_results(predicted_combined_values, model_folder)

    print(f"\nHighest Percentage: {highest_percentage:.2f}%")
    print(f"Recognized object: {best_trained_folder}")

    update_feedback(model_folder, best_trained_folder, predicted_combined_values)
