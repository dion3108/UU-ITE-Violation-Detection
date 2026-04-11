import csv

input_file = r'f:\kuliah\uu-ite\Revisi\dataset_preprocessed_with_stopwords_normalized.csv'
output_file = r'f:\kuliah\uu-ite\Revisi\dataset_preprocessed_with_stopwords_normalized_cleaned.csv'

# Baca data dan filter baris yang tidak kosong
print("Membaca file CSV...")
valid_rows = []
removed_rows = []

with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    
    for row in reader:
        clean_empty = not row['clean_teks'] or row['clean_teks'].strip() == ''
        normal_empty = not row['normal_teks'] or row['normal_teks'].strip() == ''
        
        if clean_empty or normal_empty:
            removed_rows.append(row['Indeks'])
        else:
            valid_rows.append(row)

print(f"\nTotal baris awal: {len(valid_rows) + len(removed_rows)}")
print(f"Baris yang dihapus: {len(removed_rows)}")
print(f"Baris yang tersisa: {len(valid_rows)}")

# Simpan hasil
print(f"\nMenyimpan hasil ke {output_file}...")
with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(valid_rows)

print("\n" + "="*80)
print("SELESAI!")
print("="*80)
print(f"File output: {output_file}")
print(f"Total baris di file baru: {len(valid_rows)}")
print("="*80)

# Tampilkan indeks yang dihapus
print(f"\nIndeks yang dihapus ({len(removed_rows)} baris):")
print(", ".join(removed_rows))
