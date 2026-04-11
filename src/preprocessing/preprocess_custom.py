import pandas as pd
import re
from unidecode import unidecode
import emoji

# Load dataset
df = pd.read_csv('Data/gabungan_komentar_combined_filled.csv')

# Keep original comments
df['Komentar'] = df['Komentar'].copy()  # Original column

# 1. Case Folding
df['Komentar_bersih'] = df['Komentar'].str.lower()

# 2. Demojize emojis
def demojize_text(text):
    # Add spaces around emojis using regex
    emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]'
    text = re.sub(emoji_pattern, r' \g<0> ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Then demojize
    demojized = emoji.demojize(text)
    # Replace :emoji: with EMOJI_emoji
    demojized = re.sub(r':([^:]+):', r'EMOJI_\1', demojized)
    return demojized

df['Komentar_bersih'] = df['Komentar_bersih'].apply(demojize_text)

# 3. Normalize special characters to regular ASCII
df['Komentar_bersih'] = df['Komentar_bersih'].apply(lambda x: unidecode(x))

# 4. Split compound words (e.g., AErO88menawarkan -> AErO88 menawarkan)
def split_compound_words(text):
    # Pattern: letters + digits + letters
    return re.sub(r'([a-zA-Z]+)(\d+)([a-zA-Z]+)', r'\1 \2 \3', text)

df['Komentar_bersih'] = df['Komentar_bersih'].apply(split_compound_words)

# 5. Handle URLs: Replace with LINK0 if not already tokenized
def replace_urls(text):
    # Find URLs and replace with LINK0
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(url_pattern, 'LINK0', text)

df['Komentar_bersih'] = df['Komentar_bersih'].apply(replace_urls)

# 6. Extract mentions from Label_Validator == 1.0 for judi
judi_mentions = set()
for text in df[df['Label_Validator'] == 1.0]['Komentar_bersih']:
    mentions = re.findall(r'@(\w+)', text)
    judi_mentions.update(mentions)

print("Mentions from Label_Validator=1 (Judi):", list(judi_mentions)[:10])  # Sample first 10

# Save judi mentions to CSV
judi_df = pd.DataFrame({'Mention': list(judi_mentions)})
judi_df.to_csv('Data/judi_mentions.csv', index=False)
print(f"Saved {len(judi_mentions)} judi mentions to Data/judi_mentions.csv")

# Analyze patterns for rule-based detection
# Common patterns: ends with numbers like 789, or contains 'hoki', etc.
patterns = [r'\d{3}$', r'hoki', r'parlay']  # Example patterns based on sample
judi_patterns = re.compile('|'.join(patterns))

def is_judi_mention(username):
    return username in judi_mentions or judi_patterns.search(username)

# 7. Handle Mentions
def replace_mentions(text):
    def replacer(match):
        username = match.group(1)
        if is_judi_mention(username):
            return f'MENTION_{username}'
        else:
            return 'MENTION_USER'
    return re.sub(r'@(\w+)', replacer, text)

df['Komentar_bersih'] = df['Komentar_bersih'].apply(replace_mentions)

# 8. Remove Hashtags
df['Komentar_bersih'] = df['Komentar_bersih'].apply(lambda x: re.sub(r'#\w+', '', x))

# 6. Handle Emojis (optional, as per previous suggestions)
# df['Komentar'] = df['Komentar'].apply(lambda x: emoji.replace_emoji(x, ''))

# Save processed dataset with selected columns
df[['Komentar', 'Komentar_bersih', 'Label', 'Label_Validator']].to_csv('Data/preprocessed_dataset.csv', index=False)

print("Preprocessing completed. Saved to Data/preprocessed_dataset.csv")
print("Sample processed comments:")
print(df['Komentar_bersih'].head(10))
