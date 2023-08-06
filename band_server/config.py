import json

def get_config(json_path):
    with open(json_path, "r") as f:
        config = json.load(f)
    return config["port"], config["models"]