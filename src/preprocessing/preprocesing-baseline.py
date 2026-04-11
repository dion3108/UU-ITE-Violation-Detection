import pandas as pd
import re
import string
# Gunakan Sastrawi untuk Stopword Bahasa Indonesia
# Instal jika belum ada: pip install Sastrawi
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

def preprocess_uuite_full(text):
    # 1. Lowercase
    text = str(text).lower()
    
    # 2. Hapus URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Hapus simbol @ dan # (kata dipertahankan)
    text = re.sub(r'@(\w+)', r'\1', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # 4. Hapus semua tanda baca
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # 5. Hapus semua angka
    text = re.sub(r'\d+', '', text)
    
    # 6. Stopword Removal menggunakan Sastrawi
    factory = StopWordRemoverFactory()
    # Anda juga bisa menambahkan stopword kustom jika diperlukan
    # more_stopwords = ["yg", "sdh", "dlm"]
    # stopword_remover = factory.create_stop_word_remover(more_stopwords)
    stopword_remover = factory.create_stop_word_remover()
    text = stopword_remover.remove(text)
    
    # 7. Hapus karakter Non-ASCII (Emoji)
    text = text.encode("ascii", "ignore").decode("utf-8")
    
    # 8. Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Load dataset asli
df = pd.read_csv("dataset_no_duplicates.csv")

print("🔄 Sedang memproses teks + Stopword Removal...")
df["clean_teks"] = df["Komentar"].astype(str).apply(preprocess_uuite_full)

# Atur urutan kolom yang diinginkan
column_order = ["Indeks", "Komentar", "clean_teks", "Label", "Label_Validator"]
df = df[column_order]

# Simpan ke file baru
df.to_csv("dataset_preprocessed_with_stopwords.csv", index=False)
print("✅ Selesai! Kolom 'clean_teks' kini sudah bebas dari stopword.")
print(f"📋 Urutan kolom: {', '.join(column_order)}")