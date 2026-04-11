import csv

input_file = r'f:\kuliah\uu-ite\Revisi\dataset_preprocessed_with_stopwords_normalized.csv'

# Baca data dan cari baris dengan nilai kosong
print("Mencari baris dengan nilai kosong...")
empty_rows = []

with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        clean_empty = not row['clean_teks'] or row['clean_teks'].strip() == ''
        normal_empty = not row['normal_teks'] or row['normal_teks'].strip() == ''
        
        if clean_empty or normal_empty:
            empty_rows.append({
                'Indeks': row['Indeks'],
                'Komentar': row['Komentar'],
                'clean_teks': row['clean_teks'],
                'normal_teks': row['normal_teks'],
                'clean_empty': clean_empty,
                'normal_empty': normal_empty
            })

print("\n" + "="*80)
print(f"BARIS DENGAN NILAI KOSONG: {len(empty_rows)} baris")
print("="*80)

if empty_rows:
    print("\nDaftar Indeks yang kosong:")
    print("-"*80)
    
    indeks_list = [row['Indeks'] for row in empty_rows]
    print(f"Indeks: {', '.join(indeks_list)}")
    
    print("\n" + "-"*80)
    print("Detail baris yang kosong:")
    print("-"*80)
    
    for item in empty_rows[:20]:  # Tampilkan 20 baris pertama
        print(f"\nIndeks {item['Indeks']}:")
        print(f"  Komentar: {item['Komentar']}")
        print(f"  clean_teks: '{item['clean_teks']}' (kosong: {item['clean_empty']})")
        print(f"  normal_teks: '{item['normal_teks']}' (kosong: {item['normal_empty']})")
    
    if len(empty_rows) > 20:
        print(f"\n... dan {len(empty_rows) - 20} baris lainnya")
else:
    print("\nTidak ada baris dengan nilai kosong.")

print("\n" + "="*80)
