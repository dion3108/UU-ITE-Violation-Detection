import csv
import re

# Load kamus slang
print("Loading kamus slang...")
slang_dict = {}
with open(r'f:/kuliah/uu-ite/Revisi/kamus-slang.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        slang_dict[row['slang'].lower()] = row['baku']

print(f"Total kata slang dalam kamus: {len(slang_dict)}")

# Fungsi untuk normalisasi slang
def normalize_slang(text):
    if not text or text.strip() == '':
        return text
    
    words = text.split()
    normalized_words = []
    
    for word in words:
        # Cek apakah kata ada di kamus slang (case insensitive)
        word_lower = word.lower()
        if word_lower in slang_dict:
            normalized_words.append(slang_dict[word_lower])
        else:
            normalized_words.append(word)
    
    return ' '.join(normalized_words)

# Proses dataset
input_file = r'f:/kuliah/uu-ite/Revisi/dataset_preprocessed_with_stopwords.csv'
output_file = r'f:/kuliah/uu-ite/Revisi/dataset_preprocessed_with_stopwords_normalized.csv'

print("\nMemproses dataset...")
data = []
total_normalized = 0

with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    
    for idx, row in enumerate(reader, 1):
        if idx % 5000 == 0:
            print(f"Memproses baris {idx}...")
        
        clean_teks = row['clean_teks']
        normal_teks = normalize_slang(clean_teks)
        
        # Hitung berapa baris yang dinormalisasi
        if clean_teks != normal_teks:
            total_normalized += 1
        
        # Tambahkan kolom normal_teks
        row['normal_teks'] = normal_teks
        data.append(row)

# Urutan kolom yang diinginkan
fieldnames = ['Indeks', 'Komentar', 'clean_teks', 'normal_teks', 'Label', 'Label_Validator']

# Simpan hasil
print(f"\nMenyimpan hasil ke {output_file}...")
with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

print("\n" + "="*80)
print("SELESAI!")
print("="*80)
print(f"Total baris diproses: {len(data)}")
print(f"Total baris yang dinormalisasi: {total_normalized}")
print(f"File output: {output_file}")
print("="*80)

# Tampilkan contoh normalisasi
print("\nContoh 10 baris yang dinormalisasi:")
print("-"*80)
count = 0
for row in data:
    if row['clean_teks'] != row['normal_teks'] and count < 10:
        print(f"Indeks {row['Indeks']}:")
        print(f"  clean_teks: {row['clean_teks']}")
        print(f"  normal_teks: {row['normal_teks']}")
        print()
        count += 1
