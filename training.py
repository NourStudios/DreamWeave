import os
from PIL import Image

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

def train_model(image_folder, model_folder, learned_name):
    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            img = Image.open(image_path)
            width, height = img.size

            combined_values = []

            for y in range(height):
                for x in range(width):
                    pixel = img.getpixel((x, y))
                    combined_value = get_combined_value(pixel)
                    combined_values.append(combined_value)

            result_path = os.path.join(model_folder, f"{filename}_{learned_name}_results.txt")
            with open(result_path, "w") as result_file:
                for i, combined_value in enumerate(combined_values):
                    result_file.write(f"Pixel {i + 1}: Combined Value = {combined_value:.2f}\n")

if __name__ == "__main__":?
    image_folder = input("Enter the folder containing the training images: ")
    model_folder = input("Enter the model folder: ")
    learned_name = input("Enter the name of what it has learned: ")

    train_model(image_folder, model_folder, learned_name)
