import pandas as pd

# Baca CSV file
input_file = 'Indobert - cnn-ori.csv'
output_file = 'Indobert-cnn-ori-custom-sorted.csv'

df = pd.read_csv(input_file)

# Konversi kolom persentase ke numerik
percentage_columns = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
for col in percentage_columns:
    if col in df.columns:
        df[col] = df[col].str.rstrip('%').astype('float')

# ============================================================
# KONFIGURASI: Tentukan urutan kolom yang Anda inginkan
# ============================================================

# 1. Urutan kolom untuk SORTING (prioritas dari kiri ke kanan)
#    Kolom paling kiri = prioritas tertinggi
sort_columns = [
    'batch_size',        # Prioritas 14
    'conv_filters',      # Prioritas 5
    'kernel_size',       # Prioritas 8
    'lstm_units',        # Prioritas 6
    'dense_units',       # Prioritas 7
    # 'test_f1',           # Prioritas 1
    # 'test_accuracy',     # Prioritas 2
    # 'test_precision',    # Prioritas 3
    # 'test_recall',       # Prioritas 4
]

# 2. Urutan kolom untuk OUTPUT CSV (urutan tampilan di hasil)
#    Tentukan urutan kolom seperti apa yang Anda inginkan di CSV hasil
output_columns = [
    'experiment_id',
    'batch_size',
    'conv_filters',
    'kernel_size',
    'lstm_units',
    'dense_units',
    'drop_embed',
    'drop_conv1',
    'drop_conv2',
    'drop_bilstm',
    'drop_dense',
    'test_accuracy',
    'test_precision',
    'test_recall',
    'test_f1',
    'Unnamed: 15',
]

# ============================================================

# Filter hanya kolom yang ada di dataframe
sort_columns = [col for col in sort_columns if col in df.columns]
output_columns = [col for col in output_columns if col in df.columns]

print("="*60)
print("KONFIGURASI SORTING")
print("="*60)
print(f"\nKolom untuk sorting (prioritas dari tertinggi ke terendah):")
for i, col in enumerate(sort_columns, 1):
    print(f"  {i}. {col}")

print(f"\nUrutan kolom di output CSV:")
for i, col in enumerate(output_columns, 1):
    print(f"  {i}. {col}")

# Lakukan stable sorting dari kanan ke kiri
# sehingga kolom paling kiri menjadi prioritas tertinggi
df_sorted = df.copy()
for col in reversed(sort_columns):
    df_sorted = df_sorted.sort_values(by=col, ascending=True, kind='stable')

# Konversi kembali kolom persentase ke format string dengan %
for col in percentage_columns:
    if col in df_sorted.columns:
        df_sorted[col] = df_sorted[col].apply(lambda x: f"{x}%")

# Atur ulang urutan kolom sesuai output_columns
df_sorted = df_sorted[output_columns]

# Simpan hasil
df_sorted.to_csv(output_file, index=False)

print("="*60)
print(f"✓ File berhasil disortir dan disimpan ke: {output_file}")
print(f"✓ Total baris: {len(df_sorted)}")
print("="*60)
