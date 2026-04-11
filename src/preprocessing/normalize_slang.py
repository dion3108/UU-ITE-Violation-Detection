import pandas as pd
import re

# Load dataset yang sudah dipreprocessing
df = pd.read_csv('Data/preprocessed_dataset.csv')

# Load kamus slang
slang_dict = pd.read_csv('Data/kamus-slang.csv')

# Buat dictionary mapping slang -> normal
# Asumsikan kolom pertama adalah slang, kolom kedua adalah normal
slang_mapping = dict(zip(slang_dict.iloc[:, 0], slang_dict.iloc[:, 1]))

print("Jumlah mapping slang:", len(slang_mapping))
print("Contoh mapping:")
for i, (slang, normal) in enumerate(list(slang_mapping.items())[:5]):
    print(f"  {slang} -> {normal}")

# Fungsi untuk normalisasi slang
def normalize_slang(text):
    if pd.isna(text):
        return ""  # return empty string for NaN so result is always str

    # ensure we operate on a string
    text = str(text)
    words = text.split()
    normalized_words = []

    for word in words:
        # Cek apakah kata ada di kamus slang
        if word in slang_mapping:
            replacement = slang_mapping[word]
            # If replacement is missing or not a string (NaN/float), fallback to original word
            if pd.isna(replacement) or not isinstance(replacement, str):
                normalized_words.append(word)
            else:
                normalized_words.append(replacement)
        else:
            normalized_words.append(word)

    return ' '.join(normalized_words)

# Terapkan normalisasi slang
print("\nMelakukan normalisasi slang...")
df['normal_teks'] = df['Komentar_bersih'].apply(normalize_slang)

# Reorder columns as requested: Komentar,Komentar_bersih,normal_teks,Label,Label_Validator
desired_cols = ['Komentar', 'Komentar_bersih', 'normal_teks', 'Label', 'Label_Validator']

# Keep only columns that exist in the dataframe and in the desired order
cols_to_save = [c for c in desired_cols if c in df.columns]

# If any desired columns are missing, append the rest of the dataframe's columns to preserve data
missing = [c for c in df.columns if c not in cols_to_save]
cols_to_save.extend(missing)

# Simpan hasil dengan urutan kolom yang disesuaikan
df.to_csv('Data/preprocessed_dataset_with_slang_normalization.csv', index=False, columns=cols_to_save)

print("✅ Normalisasi slang selesai!")
print(f"📊 Total data: {len(df)}")
print("💾 File tersimpan: Data/preprocessed_dataset_with_slang_normalization.csv")

# Tampilkan contoh hasil
print("\n🔍 Contoh hasil normalisasi:")
sample_df = df[['Komentar_bersih', 'normal_teks']].head(10)
for idx, row in sample_df.iterrows():
    if row['Komentar_bersih'] != row['normal_teks']:
        print(f"Original: {row['Komentar_bersih']}")
        print(f"Normalized: {row['normal_teks']}")
        print("---")
        break  # Hanya tampilkan satu contoh yang berubah