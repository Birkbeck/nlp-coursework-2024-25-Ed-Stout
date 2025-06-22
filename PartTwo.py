import pandas as pd
from pathlib import Path

def preprocess_speeches(csv_path=None):
    """
    Reads the hansard40000.csv dataset and create df
    """
    csv_path = Path.cwd() / "texts" / "hansard40000.csv"

    df = pd.read_csv(csv_path)

    df['party'] = df['party'].replace('Labour (Co-op)', 'Labour') #1

    party_counts = df['party'].value_counts() #2a
    print("Counts per party before filtering:")
    print(party_counts)

    sorted_counts = party_counts.sort_values(ascending=False) #2b
    print("\nSorted counts (highest to lowest):")
    print(sorted_counts)

    top_parties = list(sorted_counts.index[:4]) #2c
    print(f"\nTop 4 parties: {top_parties}")

    df = df[df['party'].isin(top_parties)] #2d

    df = df.drop(columns=['Speaker']) #3

    cleaned_rows = []
    for _, row in df.iterrows(): #4
        is_speech = (row['speech_class'] == 'Speech')
        long_enough = (len(row['speech']) >= 1000)
        if is_speech and long_enough:
            cleaned_rows.append(row)

    cleaned_df = pd.DataFrame(cleaned_rows).reset_index(drop=True)
    return cleaned_df
