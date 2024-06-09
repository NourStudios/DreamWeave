import subprocess

def show_options():
    print("Enter 1 for training or 2 for prediction:")
    choice = int(input())
    if choice == 1:
        subprocess.run(["python", "scripts/training.py"])
    elif choice == 2:
        subprocess.run(["python", "scripts/prediction.py"])
    else:
        print("Invalid input. Please enter 1 or 2.")

show_options()