import argparse
import pickle
import json

def list_pickle_keys(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

        # Assuming the data is a dictionary, you can list its keys
        if isinstance(data, dict):
            keys = list(data.keys())
            result = {'names': keys}
            print(json.dumps(result))
        else:
            print("The data in the .pkl file is not a dictionary.")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List keys in a .pkl file")
    parser.add_argument("file_path", type=str, help="Path to the .pkl file")

    args = parser.parse_args()
    list_pickle_keys(args.file_path)
