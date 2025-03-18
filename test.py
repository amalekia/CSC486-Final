import json

def read_attributes(file_path = 'pad.json'):
    """
    Reads a JSON file and returns a dictionary containing the values
    of attributes P, A, and D.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: A dictionary containing the values of P, A, and D.
              Returns an empty dictionary if the file cannot be read or
              if any attribute is missing.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            attributes = {
                'P': data.get('P'),
                'A': data.get('A'),
                'D': data.get('D')
            }
            return attributes
    except (FileNotFoundError, json.JSONDecodeError, AttributeError):
        return {}


# loop for 2 minutes and print out the values in the file every 10 seconds:
import time
for i in range(12):
    print(read_attributes('pad.json'))
    time.sleep(10)


