
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Project folders
PROJECT_FOLDER = Path(__file__).parent
DATA_FOLDER = PROJECT_FOLDER / "data"
DATA_FILE = DATA_FOLDER / "shopsense_dataset.csv"


def main():

    # Create data folder
    DATA_FOLDER.mkdir(exist_ok=True)

    print("Downloading ShopSense dataset...")

    # Download dataset from UCI
    dataset = fetch_ucirepo(id=468)

    # Get input features and target
    features = dataset.data.features
    target = dataset.data.targets

    # Combine both into one dataframe
    data = pd.concat([features, target], axis=1)

    # Save dataset
    data.to_csv(DATA_FILE, index=False)

    print("Dataset downloaded successfully!")
    print("Total records:", len(data))
    print("Saved at:", DATA_FILE)


if __name__ == "__main__":
    main()
