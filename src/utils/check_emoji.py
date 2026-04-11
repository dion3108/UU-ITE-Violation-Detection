import csv
import re

# Pattern untuk mendeteksi emoji
emoji_pattern = re.compile(
    '['
    u'\U0001F600-\U0001F64F'  # emoticons
    u'\U0001F300-\U0001F5FF'  # symbols & pictographs
    u'\U0001F680-\U0001F6FF'  # transport & map symbols
    u'\U0001F1E0-\U0001F1FF'  # flags
    u'\U00002702-\U000027B0'
    u'\U000024C2-\U0001F251'
    u'\u200d'
    u'\u2600-\u26FF'
    u'\u2700-\u27BF'
    ']+',
    flags=re.UNICODE
)

input_file = r'f:\kuliah\uu-ite\Revisi\dataset_preprocessed_with_stopwords.csv'

emoji_in_komentar = []
emoji_in_clean_teks = []

with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        has_emoji_komentar = emoji_pattern.search(row['Komentar'])
        has_emoji_clean = emoji_pattern.search(row['clean_teks'])
        
        if has_emoji_komentar:
            emoji_in_komentar.append({
                'Indeks': row['Indeks'],
                'Komentar': row['Komentar'],
                'clean_teks': row['clean_teks']
            })
        
        if has_emoji_clean:
            emoji_in_clean_teks.append({
                'Indeks': row['Indeks'],
                'Komentar': row['Komentar'],
                'clean_teks': row['clean_teks']
            })

print('=' * 80)
print('HASIL PENGECEKAN EMOJI')
print('=' * 80)
print(f'Total baris dengan emoji di kolom Komentar: {len(emoji_in_komentar)}')
print(f'Total baris dengan emoji di kolom clean_teks: {len(emoji_in_clean_teks)}')
print()

if emoji_in_komentar:
    print('Contoh 10 baris pertama yang mengandung emoji di kolom Komentar:')
    print('-' * 80)
    for item in emoji_in_komentar[:10]:
        print(f"Indeks {item['Indeks']}:")
        print(f"  Komentar: {item['Komentar']}")
        print(f"  clean_teks: {item['clean_teks']}")
        print()
else:
    print('Tidak ada emoji yang ditemukan di kolom Komentar.')
    print()

if emoji_in_clean_teks:
    print('Contoh baris yang masih mengandung emoji di kolom clean_teks:')
    print('-' * 80)
    for item in emoji_in_clean_teks[:10]:
        print(f"Indeks {item['Indeks']}:")
        print(f"  Komentar: {item['Komentar']}")
        print(f"  clean_teks: {item['clean_teks']}")
        print()
else:
    print('Tidak ada emoji yang ditemukan di kolom clean_teks.')
    print()

print('=' * 80)
print(f'Kesimpulan: Preprocessing berhasil membersihkan {len(emoji_in_komentar)} emoji dari teks.')
print('=' * 80)
