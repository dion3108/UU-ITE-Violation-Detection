import pandas as pd
import re
import emoji

def cleansing_case_folding(text):
    """
    Tahap 1: Cleansing - Menghapus simbol, emoji, tanda baca, dan angka
    Tahap 2: Case Folding - Mengubah ke huruf kecil
    """
    if pd.isna(text):
        return ""
    
    # Convert to string
    text = str(text)
    
    # Remove emojis
    text = emoji.replace_emoji(text, replace='')
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation and special characters, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Convert to lowercase (Case Folding)
    text = text.lower()
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalization(text, slang_dict):
    """
    Tahap 3: Normalization - Mengganti kata tidak baku menggunakan kamus slang
    """
    if pd.isna(text) or text == "":
        return ""
    
    # Split text into words
    words = text.split()
    
    # Replace each word if it exists in slang dictionary
    normalized_words = []
    for word in words:
        replacement = slang_dict.get(word, word)
        # Ensure replacement is string, handle NaN and float values
        if pd.isna(replacement):
            normalized_words.append(word)
        else:
            normalized_words.append(str(replacement))
    
    # Join words back
    normalized_text = ' '.join(normalized_words)
    
    return normalized_text

def main():
    print("=" * 60)
    print("PREPROCESSING DATA")
    print("=" * 60)
    
    # Load dataset
    print("\n1. Loading dataset...")
    df = pd.read_csv('dataset_no_duplicates.csv')
    print(f"   Dataset loaded: {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")
    
    # Load slang dictionary
    print("\n2. Loading kamus slang...")
    slang_df = pd.read_csv('kamus-slang.csv')
    
    # Convert slang dictionary to dict format
    # Assuming the CSV has columns like 'slang' and 'formal' or similar
    # Adjust column names based on your actual CSV structure
    if slang_df.shape[1] >= 2:
        # Clean the dataframe: drop rows with NaN and convert to string
        slang_df_clean = slang_df.dropna(subset=[slang_df.columns[0], slang_df.columns[1]])
        slang_df_clean.iloc[:, 0] = slang_df_clean.iloc[:, 0].astype(str).str.strip().str.lower()
        slang_df_clean.iloc[:, 1] = slang_df_clean.iloc[:, 1].astype(str).str.strip().str.lower()
        slang_dict = dict(zip(slang_df_clean.iloc[:, 0], slang_df_clean.iloc[:, 1]))
    else:
        print("   Warning: Slang dictionary format not recognized. Using empty dict.")
        slang_dict = {}
    
    print(f"   Slang dictionary loaded: {len(slang_dict)} entries")
    
    # Tahap 1 & 2: Cleansing and Case Folding
    print("\n3. Applying Cleansing & Case Folding...")
    df['clean_teks'] = df['Komentar'].apply(cleansing_case_folding)
    print("   ✓ Column 'clean_teks' created")
    
    # Tahap 3: Normalization
    print("\n4. Applying Normalization...")
    df['normal_teks'] = df['clean_teks'].apply(lambda x: normalization(x, slang_dict))
    print("   ✓ Column 'normal_teks' created")
    
    # Reorder columns
    print("\n5. Reordering columns...")
    columns_order = ['Indeks', 'Komentar', 'clean_teks', 'normal_teks', 'Label', 'Label_Validator']
    df = df[columns_order]
    print(f"   ✓ Columns ordered: {columns_order}")
    
    # Save preprocessed dataset
    output_file = 'dataset_preprocessed.csv'
    print(f"\n6. Saving preprocessed dataset to '{output_file}'...")
    df.to_csv(output_file, index=False)
    print(f"   ✓ File saved successfully!")
    
    # Display sample results
    print("\n" + "=" * 60)
    print("SAMPLE RESULTS (First 3 rows)")
    print("=" * 60)
    for i in range(min(3, len(df))):
        print(f"\nRow {i+1}:")
        print(f"  Original  : {df.iloc[i]['Komentar'][:80]}...")
        print(f"  Clean     : {df.iloc[i]['clean_teks'][:80]}...")
        print(f"  Normalized: {df.iloc[i]['normal_teks'][:80]}...")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED!")
    print("=" * 60)
    print(f"\nOutput file: {output_file}")
    print(f"Total rows processed: {len(df)}")

if __name__ == "__main__":
    main()
