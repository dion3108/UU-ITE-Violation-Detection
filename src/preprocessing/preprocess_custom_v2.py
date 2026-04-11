import pandas as pd
import re
from unidecode import unidecode
import emoji

# Load dataset
df = pd.read_csv('Data/datasethoaks.csv')

# Keep original comments
df['Komentar'] = df['Komentar'].copy()  # Original column

# 1. Case Folding
df['Komentar_bersih'] = df['Komentar'].str.lower()

# 2. Demojize emojis
def demojize_text(text):
    # Use emoji library to detect all emojis and add spaces around them
    for em in emoji.emoji_list(text):
        text = text.replace(em['emoji'], f" {em['emoji']} ")
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Then demojize
    demojized = emoji.demojize(text)
    # Replace :emoji: with emoji_emoji
    demojized = re.sub(r':([^:]+):', r'emoji_\1', demojized)
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
    # Find URLs and replace with link0
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(url_pattern, 'link0', text)

df['Komentar_bersih'] = df['Komentar_bersih'].apply(replace_urls)

# 6. Extract mentions from Label_Validator == 1.0 for judi
judi_mentions = set()
for text in df[df['Label_Validator'] == 1.0]['Komentar_bersih']:
    mentions = re.findall(r'@(\w+)', text)
    judi_mentions.update([m.lower() for m in mentions])

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
        username = match.group(1).lower()
        if is_judi_mention(username):
            return f'mention_{username}'
        else:
            return 'mention_user'
    return re.sub(r'@(\w+)', replacer, text)

df['Komentar_bersih'] = df['Komentar_bersih'].apply(replace_mentions)

# 8. Remove Hashtags
df['Komentar_bersih'] = df['Komentar_bersih'].apply(lambda x: re.sub(r'#\w+', '', x))

# 9. Final case folding to ensure all text is lowercase
df['Komentar_bersih'] = df['Komentar_bersih'].str.lower()

# 6. Handle Emojis (optional, as per previous suggestions)
# df['Komentar'] = df['Komentar'].apply(lambda x: emoji.replace_emoji(x, ''))

# Save processed dataset with selected columns
# --- Slang normalization integration ---
# Load kamus slang if available and build mapping
try:
    slang_dict = pd.read_csv('Data/kamus-slang.csv')
    slang_mapping = dict(zip(slang_dict.iloc[:, 0].astype(str), slang_dict.iloc[:, 1].astype(str)))
    print(f"Loaded slang dictionary with {len(slang_mapping)} entries")
except Exception as e:
    slang_mapping = {}
    print(f"Warning: could not load slang dictionary: {e}")

def normalize_slang_text(text):
    # Ensure text is string
    if pd.isna(text):
        return ""
    text = str(text)
    words = text.split()
    normalized = []
    for w in words:
        # if exact match in mapping, replace; otherwise keep
        if w in slang_mapping:
            repl = slang_mapping[w]
            if pd.isna(repl) or not isinstance(repl, str):
                normalized.append(w)
            else:
                normalized.append(repl)
        else:
            normalized.append(w)
    return ' '.join(normalized)

# Apply normalization and save full preprocessed dataset including normal_teks
print("Applying slang normalization...")
df['normal_teks'] = df['Komentar_bersih'].apply(normalize_slang_text)

# Save processed dataset with desired column order
cols_to_save = ['Komentar', 'Komentar_bersih', 'normal_teks', 'Label', 'Label_Validator']
cols_to_save = [c for c in cols_to_save if c in df.columns]
other_cols = [c for c in df.columns if c not in cols_to_save]
cols_to_save.extend(other_cols)
df.to_csv('Data/preprocessed_dataset.csv', index=False, columns=cols_to_save)

print("Preprocessing + slang normalization completed. Saved to Data/preprocessed_dataset.csv")
print("Sample processed comments:")
print(df[['Komentar_bersih','normal_teks']].head(10))
