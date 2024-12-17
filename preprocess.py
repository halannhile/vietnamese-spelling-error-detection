import json
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load the VSEC dataset from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def split_data(data, test_size=0.2, dev_size=0.1):
    """Split the data into train, dev, and test sets."""
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    train_data, dev_data = train_test_split(train_data, test_size=dev_size, random_state=42)
    return train_data, dev_data, test_data

def save_data(data, output_path):
    """Save data to a JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    input_path = "./data/VSEC.jsonl"
    output_dir = "./data/"

    # Load data and split
    data = load_data(input_path)
    train, dev, test = split_data(data)

    # Save splits
    save_data(train, f"{output_dir}/train.jsonl")
    save_data(dev, f"{output_dir}/dev.jsonl")
    save_data(test, f"{output_dir}/test.jsonl")
