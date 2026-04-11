import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Create stemmer once (global)
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming_text(text):
    """
    Tahap 4: Stemming - Mengubah kata ke bentuk dasar
    """
    if pd.isna(text) or text == "":
        return ""
    
    # Convert to string
    text = str(text)
    
    # Stem the text
    stemmed_text = stemmer.stem(text)
    
    return stemmed_text

def main():
    print("=" * 60)
    print("PREPROCESSING DATA - STEMMING")
    print("=" * 60)
    
    # Load preprocessed dataset
    print("\n1. Loading preprocessed dataset...")
    df = pd.read_csv('dataset_preprocessed.csv')
    print(f"   Dataset loaded: {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")
    
    # Tahap 4: Stemming
    print("\n2. Applying Stemming on 'normal_teks' column...")
    print("   This may take a while for large datasets...")
    
    # Use apply for better performance with progress tracking
    import numpy as np
    from tqdm import tqdm
    tqdm.pandas(desc="   Processing")
    
    df['stemming_teks'] = df['normal_teks'].progress_apply(stemming_text)
    
    print("   ✓ Column 'stemming_teks' created")
    
    # Reorder columns
    print("\n3. Reordering columns...")
    columns_order = ['Indeks', 'Komentar', 'clean_teks', 'normal_teks', 'stemming_teks', 'Label', 'Label_Validator']
    df = df[columns_order]
    print(f"   ✓ Columns ordered: {columns_order}")
    
    # Save preprocessed dataset
    output_file = 'dataset_preprocessed_stemmed.csv'
    print(f"\n4. Saving preprocessed dataset to '{output_file}'...")
    df.to_csv(output_file, index=False)
    print(f"   ✓ File saved successfully!")
    
    # Display sample results
    print("\n" + "=" * 60)
    print("SAMPLE RESULTS (First 3 rows)")
    print("=" * 60)
    for i in range(min(3, len(df))):
        print(f"\nRow {i+1}:")
        print(f"  Original  : {df.iloc[i]['Komentar'][:60]}...")
        print(f"  Clean     : {df.iloc[i]['clean_teks'][:60]}...")
        print(f"  Normalized: {df.iloc[i]['normal_teks'][:60]}...")
        print(f"  Stemmed   : {df.iloc[i]['stemming_teks'][:60]}...")
    
    print("\n" + "=" * 60)
    print("STEMMING COMPLETED!")
    print("=" * 60)
    print(f"\nOutput file: {output_file}")
    print(f"Total rows processed: {len(df)}")

if __name__ == "__main__":
    main()
