import pandas as pd
from tqdm import tqdm
import g4f

# === 1. Load data slang ===
df = pd.read_csv("Data/candidate_slang.csv")

# === 2. Fungsi normalisasi pakai g4f ===
def normalize_with_g4f(word):
    try:
        response = g4f.ChatCompletion.create(
            model="gpt-4",  # bisa juga gpt-3.5 atau model lain yg tersedia
            messages=[{
                "role": "user",
                "content": f"Normalisasikan kata slang atau typo berikut menjadi bahasa Indonesia baku: '{word}'. "
                           f"Jika sudah baku, kembalikan apa adanya. Jawab hanya kata hasilnya."
            }]
        )
        return response
    except Exception as e:
        return None

# === 3. Jalankan normalisasi dengan progress bar ===
corrected_words = []
for slang in tqdm(df["slang_word"], desc="🔄 Normalisasi dengan g4f"):
    corrected_words.append(normalize_with_g4f(slang))

df["correct_word"] = corrected_words

# === 4. Simpan hasil ===
df.to_csv("candidate_slang_normalized_g4f.csv", index=False, encoding="utf-8")

print("✅ Normalisasi selesai dengan g4f!")
print("📂 Hasil tersimpan di candidate_slang_normalized_g4f.csv")
