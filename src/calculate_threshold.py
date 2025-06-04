import os
import json
import glob
import numpy as np
import math
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate threshold")

    parser.add_argument(
        "--model",
        type=str,
        default="llama3_8b",
    )

    args = parser.parse_args()
    return args

folder_path = "./result_app/"  
args = parse_args()
args = parse_args()
fields = ["nll2"]
data = {field: [] for field in fields}
model_name = args.model

file_list = glob.glob(os.path.join(folder_path, f"{model_name}*.jsonl"))

for file_path in file_list:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                for field in fields:
                    val = record.get(field)
                    if val is not None and not (isinstance(val, float) and math.isnan(val)):
                        data[field].append(val)
            except Exception as e:
                print(f"Error in file {file_path}: {e}")

# mean and std for NLL 
for field in fields:
    values = data[field]
    if values:
        mean = np.mean(values)
        std = np.std(values)
        print(f"{field}: {mean:.4f} ± {std:.4f} (n={len(values)})")
        result = mean+std
        print(f"Threshold for {model_name} can be set as {result}")
    else:
        print(f"{field}: No valid data found.")