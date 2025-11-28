import json
import os


def save_data(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".json"):
        with open(path, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    elif path.endswith(".txt"):
        with open(path, "w", encoding='utf-8') as f:
            f.write(data)
    else:
        raise ValueError("Only support json format")


def load_data(path):
    if path.endswith(".txt"):
        with open(path, "r", encoding='utf-8') as f:
            data = f.readlines()
    else:
        raise ValueError("Only supports txt format")
    return data
