import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('Data/gabungan_komentar_combined_filled.csv')

# Check label distribution
print("Label_Validator distribution:")
print(df['Label_Validator'].value_counts())

# Get unique labels
labels = df['Label_Validator'].unique()
num_labels = len(labels)

# Calculate sample size per label for balance
sample_per_label = 1000 // num_labels
remainder = 1000 % num_labels

# Sample stratified
sampled_df = pd.DataFrame()
for label in labels:
    label_df = df[df['Label_Validator'] == label]
    n = sample_per_label + (1 if remainder > 0 else 0)
    remainder -= 1
    if len(label_df) >= n:
        sampled = label_df.sample(n=n, random_state=42)
    else:
        sampled = label_df  # If less than n, take all
    sampled_df = pd.concat([sampled_df, sampled])

# If total less than 1000, sample more from largest
if len(sampled_df) < 1000:
    remaining = 1000 - len(sampled_df)
    largest_label = df['Label_Validator'].value_counts().idxmax()
    largest_df = df[df['Label_Validator'] == largest_label]
    additional = largest_df.sample(n=remaining, random_state=42)
    sampled_df = pd.concat([sampled_df, additional])

# Shuffle the final dataset
sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to new CSV
sampled_df.to_csv('Data/test_dataset.csv', index=False)

print(f"Created test dataset with {len(sampled_df)} samples.")
print("New label distribution:")
print(sampled_df['Label_Validator'].value_counts())
