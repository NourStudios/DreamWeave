import subprocess

def show_options():
    print("Enter 1 for text-based model or 2 for image-based model:")
    choice = int(input())
    if choice == 1:
        subprocess.run(["python", "scripts/1.py"])
    elif choice == 2:
        subprocess.run(["python", "scripts/2.py"])
    else:
        print("Invalid input. Please enter 1 or 2.")

show_options()