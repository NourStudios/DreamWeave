import os
from PIL import Image

def get_brightness(pixel):
    return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]

def predict_brightness(image_path, learned_folder):
    img = Image.open(image_path)
    width, height = img.size

    brightness_levels = []

    for y in range(height):
        for x in range(width):
            pixel = img.getpixel((x, y))
            brightness = get_brightness(pixel)
            brightness_levels.append(brightness)

    return brightness_levels

def compare_results(predicted_brightness, model_folder):
    highest_percentage = 0
    best_trained_folder = ""

    for trained_folder_name in os.listdir(model_folder):
        trained_folder_path = os.path.join(model_folder, trained_folder_name)

        if os.path.isdir(trained_folder_path):
            percentages = calculate_percentage(predicted_brightness, trained_folder_path)

            for filename, percentage in percentages:
                if percentage > highest_percentage:
                    highest_percentage = percentage
                    best_trained_folder = trained_folder_name

                print(f"{filename}: {percentage:.2f}%")

    return highest_percentage, best_trained_folder


def calculate_percentage(predicted_brightness, trained_folder):
    percentages = []

    for filename in os.listdir(trained_folder):
        if filename.endswith('_results.txt'):
            result_path = os.path.join(trained_folder, filename)

            with open(result_path, "r") as result_file:
                matched_pixels = 0
                total_pixels = 0

                for i, line in enumerate(result_file):
                    if i < len(predicted_brightness):  # Check the index before accessing the list
                        trained_brightness = float(line.split()[-1][:-1])
                        predicted_brightness_i = predicted_brightness[i]

                        if trained_brightness == predicted_brightness_i:
                            matched_pixels += 1

                        total_pixels += 1

                percentage = (matched_pixels / total_pixels) * 100
                percentages.append((filename, percentage))

    return percentages


if __name__ == "__main__":
    image_path = input("Enter the image to predict: ")
    model_folder = input("Enter the model folder: ")

    predicted_brightness = predict_brightness(image_path, model_folder)
    highest_percentage, best_trained_folder = compare_results(predicted_brightness, model_folder)

    print(f"\nHighest Percentage: {highest_percentage:.2f}%")
    print(f"Recognised object: {best_trained_folder}")
