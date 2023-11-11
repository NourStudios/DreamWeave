import os
from PIL import Image

def get_brightness(pixel):
    return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]

def show_brightness(image_path):
    img = Image.open(image_path)
    width, height = img.size

    brightness_levels = []

    for y in range(height):
        for x in range(width):
            pixel = img.getpixel((x, y))
            brightness = get_brightness(pixel)
            brightness_levels.append(brightness)

    return brightness_levels

def save_results(model_name, sample_folder, learned_name):
    model_folder = os.path.join(os.getcwd(), model_name)

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    learned_folder = os.path.join(model_folder, learned_name)

    if not os.path.exists(learned_folder):
        os.makedirs(learned_folder)

    for filename in os.listdir(sample_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(sample_folder, filename)
            brightness_levels = show_brightness(image_path)

            result_path = os.path.join(learned_folder, f"{filename}_results.txt")
            with open(result_path, "w") as result_file:
                for i, brightness in enumerate(brightness_levels):
                    result_file.write(f"Pixel {i + 1}: Brightness = {brightness:.2f}\n")

if __name__ == "__main__":
    model_name = input("Enter the model name: ")
    sample_folder = input("Enter the folder containing the sample images: ")
    learned_name = input("Enter the name of what it learned: ")

    save_results(model_name, sample_folder, learned_name)
