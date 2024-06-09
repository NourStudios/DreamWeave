import subprocess

def main():
    choice = input("Text-Based Model(1) or Image-Based Model(2): ")

    if choice == '1':
        subprocess.run(["python", "3.py"])
    elif choice == '2':
        subprocess.run(["python", "4.py"])
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
