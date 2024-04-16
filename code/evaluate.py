import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None, required=True, help="path of input")
    parser.add_argument("--output_fname", type=str, default=None, required=True, help="path of output file")
    return parser.parse_args()

def main():
    args = parse_args()

    input_path = args.path
    df = pd.read_csv(input_path)

    pred = [str(i).lower().strip() for i in df["prediction"]]
    true = [str(i).lower().strip() for i in df["label"]]

    correct = df[df["prediction"] == df["label"]]
    results = {}
    print(input_path) 
    print(f"{len(correct)}/{len(df)}", accuracy_score(true, pred))
    print(classification_report(true, pred, digits=4, zero_division=0))
    
    import json
    results['acc_score'] = accuracy_score(true, pred)
    results['classification_report'] = classification_report(true, pred, digits=4, zero_division=0)
    
    # Define the file path
    file_path = args.output_fname + ".json"

    # Open file in write mode and dump data
    with open(file_path, 'w') as json_file:
        json.dump(results, json_file)

if __name__ == "__main__":
    main()