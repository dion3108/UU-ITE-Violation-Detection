import csv
import re

input_file = r'f:\kuliah\uu-ite\Revisi\dataset_preprocessed_with_stopwords.csv'

# Baca data dan cari komentar yang memiliki angka
print("Mencari komentar yang mengandung angka...")
rows_with_numbers = []

with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        # Cek apakah Komentar asli mengandung angka
        if re.search(r'\d', row['Komentar']):
            rows_with_numbers.append({
                'Indeks': row['Indeks'],
                'Komentar': row['Komentar'],
                'clean_teks': row['clean_teks'],
                'has_number_in_clean': bool(re.search(r'\d', row['clean_teks']))
            })

print("\n" + "="*80)
print(f"PERBANDINGAN KOMENTAR DENGAN ANGKA")
print("="*80)
print(f"Total komentar yang mengandung angka: {len(rows_with_numbers)}")

# Cek apakah ada yang masih punya angka di clean_teks
still_has_numbers = [r for r in rows_with_numbers if r['has_number_in_clean']]
print(f"Komentar yang masih ada angka di clean_teks: {len(still_has_numbers)}")

print("\n" + "-"*80)
print("Contoh 20 baris pertama:")
print("-"*80)

for i, item in enumerate(rows_with_numbers[:20], 1):
    print(f"\n{i}. Indeks {item['Indeks']}:")
    print(f"   Komentar   : {item['Komentar']}")
    print(f"   clean_teks : {item['clean_teks']}")
    if item['has_number_in_clean']:
        print(f"   ⚠️ MASIH ADA ANGKA di clean_teks!")

if still_has_numbers:
    print("\n" + "="*80)
    print("⚠️ PERINGATAN: Ada komentar yang masih mengandung angka!")
    print("="*80)
    for item in still_has_numbers[:10]:
        print(f"\nIndeks {item['Indeks']}:")
        print(f"   Komentar   : {item['Komentar']}")
        print(f"   clean_teks : {item['clean_teks']}")
else:
    print("\n" + "="*80)
    print("✅ SUKSES: Semua angka berhasil dihapus dari clean_teks!")
    print("="*80)
