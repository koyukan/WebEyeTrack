import argparse
import pathlib

import pandas as pd

def main(args):
    dataset = pathlib.Path(args.dataset)

    # Load the meta file
    meta = pd.read_csv(f"{dataset}/participant_characteristics.csv")
    
    # Iterate over all the participants data
    for idx, row in meta.iterrows():
        participant = row['Participant ID']

        # Load the data
        p_dir = dataset / participant
        if not p_dir.exists():
            print(f"Participant {participant} does not exist")
            continue

        # Load the data
        print(f"Loading participant {participant} data")

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extracts the eye tracking data from the webgazer dataset')
    parser.add_argument('-d', '--dataset', required=True, type=str, help='Directory of the WebGazerETRA2018Dataset folder')
    parser.add_argument('-o', '--output', default="./output", type=str, help='Directory to save the extracted data')
    args = parser.parse_args()

    main(args)