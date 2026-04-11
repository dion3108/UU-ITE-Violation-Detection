import pandas as pd

# Baca CSV file
input_file = 'Indobert-grid_10k-sorted-verified.csv'
output_file = 'Indobert-grid_10k-multi-pass-sorted-asc.csv'

df = pd.read_csv(input_file)

# Konversi kolom persentase ke numerik
percentage_columns = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
for col in percentage_columns:
    if col in df.columns:
        df[col] = df[col].str.rstrip('%').astype('float')

# Ambil semua kolom kecuali experiment_id untuk sorting
# Urutkan sesuai urutan asli di CSV
all_columns = df.columns.tolist()
sort_columns = [col for col in all_columns if col != 'experiment_id' and col != 'Unnamed: 15']

# Untuk stable multi-pass sort, kita sort dari kanan ke kiri
# sehingga kolom paling kiri menjadi prioritas tertinggi
print(f"Kolom yang akan digunakan untuk sorting (urutan prioritas dari kiri ke kanan):")
for i, col in enumerate(sort_columns, 1):
    print(f"{i}. {col}")

# Lakukan stable sorting dari kanan ke kiri
# Pandas sort_values dengan kind='stable' (default di pandas >= 1.2.0)
df_sorted = df.copy()
for col in reversed(sort_columns):  # Reverse agar yang paling kiri jadi prioritas tertinggi
    df_sorted = df_sorted.sort_values(by=col, ascending=True, kind='stable')

# Konversi kembali kolom persentase ke format string dengan %
for col in percentage_columns:
    if col in df_sorted.columns:
        df_sorted[col] = df_sorted[col].apply(lambda x: f"{x}%")

# Simpan hasil (urutan kolom tetap seperti asli)
df_sorted.to_csv(output_file, index=False)

print(f"\nFile berhasil disortir dan disimpan ke: {output_file}")
print(f"Total baris: {len(df_sorted)}")
