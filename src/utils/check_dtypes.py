import csv
import re

input_file = r'f:\kuliah\uu-ite\Revisi\dataset_preprocessed_with_stopwords_normalized.csv'

# Baca data
print("Membaca file CSV...")
data = []
with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        data.append(row)

print(f"\nTotal baris: {len(data)}")
print(f"Total kolom: {len(fieldnames)}")
print("\n" + "="*80)
print("ANALISIS TIPE DATA KOLOM")
print("="*80)

# Fungsi untuk mengecek tipe data
def analyze_column(column_name, values):
    # Ambil sample non-empty values
    non_empty = [v for v in values if v and v.strip()]
    
    if not non_empty:
        return "Empty/Null"
    
    # Cek apakah semua integer
    all_int = all(v.isdigit() or (v.startswith('-') and v[1:].isdigit()) for v in non_empty)
    if all_int:
        return "Integer"
    
    # Cek apakah semua float
    all_float = True
    for v in non_empty:
        try:
            float(v)
        except ValueError:
            all_float = False
            break
    if all_float:
        return "Float/Numeric"
    
    # Sisanya adalah string
    return "String/Text"

# Analisis setiap kolom
for col in fieldnames:
    values = [row[col] for row in data]
    dtype = analyze_column(col, values)
    
    # Hitung statistik
    non_empty = [v for v in values if v and v.strip()]
    empty_count = len(values) - len(non_empty)
    
    print(f"\nKolom: {col}")
    print(f"  Tipe Data: {dtype}")
    print(f"  Total nilai: {len(values)}")
    print(f"  Nilai kosong: {empty_count}")
    print(f"  Nilai terisi: {len(non_empty)}")
    
    # Tampilkan contoh nilai
    if non_empty:
        samples = non_empty[:5]
        print(f"  Contoh nilai:")
        for i, sample in enumerate(samples, 1):
            # Batasi panjang tampilan
            display_val = sample if len(sample) <= 60 else sample[:57] + "..."
            print(f"    {i}. {display_val}")
    
    # Info tambahan untuk kolom numerik
    if dtype in ["Integer", "Float/Numeric"]:
        numeric_vals = [float(v) for v in non_empty]
        print(f"  Min: {min(numeric_vals)}")
        print(f"  Max: {max(numeric_vals)}")
        print(f"  Unique values: {len(set(non_empty))}")
    elif dtype == "String/Text":
        # Hitung panjang rata-rata
        avg_length = sum(len(v) for v in non_empty) / len(non_empty)
        max_length = max(len(v) for v in non_empty)
        print(f"  Panjang rata-rata: {avg_length:.1f} karakter")
        print(f"  Panjang maksimal: {max_length} karakter")
        print(f"  Unique values: {len(set(non_empty))}")

print("\n" + "="*80)
