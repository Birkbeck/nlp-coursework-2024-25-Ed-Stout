import pandas as pd
from pathlib import Path

def preprocess_speeches(csv_path=None):
    """
    Reads the hansard40000.csv dataset and create df
    """
    csv_path = Path.cwd() / "texts" / "hansard40000.csv"

    df = pd.read_csv(csv_path)

    df['party'] = df['party'].replace('Labour (Co-op)', 'Labour') #1

    # 2. Filter to the four most common parties
    # 2a. Count speeches per party
    party_counts = df['party'].value_counts()
    print("Counts per party before filtering:")
    print(party_counts)

    # 2b. Sort counts descending
    sorted_counts = party_counts.sort_values(ascending=False)
    print("\nSorted counts (highest to lowest):")
    print(sorted_counts)

    # 2c. Select the top 4 parties
    top_parties = list(sorted_counts.index[:4])
    print(f"\nTop 4 parties: {top_parties}")

    # 2d. Filter DataFrame to only top 4 parties
    df = df[df['party'].isin(top_parties)]
