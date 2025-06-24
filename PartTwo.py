import pandas as pd
from pathlib import Path

csv_path = Path.cwd() / "p2-texts" / "hansard40000.csv"

df = pd.read_csv(csv_path)

df['party'] = df['party'].replace('Labour (Co-op)', 'Labour') #1

df = df[df['party'] != 'Speaker'] #2c

party_counts = df['party'].value_counts() #2a
print("Counts per party before filtering:")
print(party_counts)

sorted_counts = party_counts.sort_values(ascending=False) #2b
print("\nSorted counts (highest to lowest):")
print(sorted_counts)

top_parties = list(sorted_counts.index[:4]) #2c
print(f"\nTop 4 parties: {top_parties}")

#df = df[df['party'].isin(top_parties)] #2d

#df = df[df['speech_class'] == 'Speech'] #3

#df = df[df['speech'].str.len() >= 1000] #4

"""cleaned_rows = []
for _, row in df.iterrows(): #4
    is_speech = (row['speech_class'] == 'Speech')
    long_enough = (len(row['speech']) >= 1000)
    if is_speech and long_enough:
        cleaned_rows.append(row)"""

#cleaned_df = pd.DataFrame(cleaned_rows).reset_index(drop=True)
#return cleaned_df

new_party_counts = df['party'].value_counts().sort_values(ascending=False) 
print(new_party_counts)
