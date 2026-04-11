import pandas as pd

# Baca file CSV
df = pd.read_csv('Indobert-grid_10k-sorted-verified.csv')

# Konversi kolom persentase ke float
percentage_columns = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
for col in percentage_columns:
    if col in df.columns:
        df[col] = df[col].str.rstrip('%').astype('float')

# Kolom numerik yang akan diurutkan (dari kanan ke kiri untuk stable sort)
# Urutan: test_f1 -> test_recall -> test_precision -> test_accuracy
sort_columns = [
    'test_f1',
    'test_recall', 
    'test_precision',
    'test_accuracy'
]

# Filter hanya kolom yang ada di dataframe
sort_columns = [col for col in sort_columns if col in df.columns]

print("Melakukan stable multi-pass sorting dari paling kecil ke paling besar...")
print(f"Urutan sorting: {' -> '.join(reversed(sort_columns))}")

# Stable multi-pass sorting dari paling kecil ke paling besar
# Sort dari kolom terakhir ke pertama untuk menjaga stabilitas
for col in reversed(sort_columns):
    df = df.sort_values(by=col, ascending=True, kind='stable')
    print(f"Sorted by {col} (ascending)")

# Konversi kembali ke format persentase untuk output
for col in percentage_columns:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: f"{x}%")

# Simpan hasil
output_file = 'Indobert-grid_10k-sorted-ascending.csv'
df.to_csv(output_file, index=False)

print(f"\nHasil disimpan ke: {output_file}")
print(f"Total baris: {len(df)}")
print("\n5 baris pertama (nilai terkecil):")
print(df.head())
