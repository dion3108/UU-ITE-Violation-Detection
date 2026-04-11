import pandas as pd
import Levenshtein
from tqdm import tqdm

# === 1. Load data slang ===
df = pd.read_csv("Data/candidate_slang.csv")

# === 2. Load kamus KBBI ===
with open("Data/kbbi.txt", "r", encoding="utf-8") as f:
    kbbi_words = [line.strip().lower() for line in f if line.strip()]
kbbi_set = set(kbbi_words)

# === 3. Mapping manual untuk slang khas ===
manual_mapping = {
    "anjirrrrrr": "anjir",
    "apankabr": "apa kabar",
    "yamg": "yang",
    "meluk": "peluk",
    "kerreeeeenn": "keren",
    "beliu": "beliau",
    "pemirikiran": "pemikiran",
    "memerhati": "memerhatikan",
    "ditutup2i": "ditutupi",
    "kaburrr": "kabur",
    "kageeett": "kaget",
    "cowo": "cowok",
    "nganjing": "anjing",
    "gorangan": "gorengan",
    "klou": "kalau",
    "auoto": "auto",
    "wleeee": "weleh",
    # tambahkan sesuai kebutuhan
}

# === 4. Fungsi koreksi hybrid ===
def correct_word(word, max_dist=2):
    w = str(word).lower()

    # 1) cek mapping manual
    if w in manual_mapping:
        return manual_mapping[w]

    # 2) cek di kamus langsung
    if w in kbbi_set:
        return w

    # 3) pakai edit distance (dengan filter)
    subset = [kw for kw in kbbi_words if abs(len(kw) - len(w)) <= max_dist and kw[0] == w[0]]

    closest_word = None
    closest_dist = 999
    for kw in subset:
        dist = Levenshtein.distance(w, kw)
        if dist < closest_dist:
            closest_dist = dist
            closest_word = kw
            if dist == 1:  # early stop
                break

    return closest_word if closest_word and closest_dist <= max_dist else None

# === 5. Terapkan dengan progress bar ===
corrected_words = []
for word in tqdm(df["slang_word"], desc="🔄 Normalisasi slang words"):
    corrected_words.append(correct_word(word))

df["correct_word"] = corrected_words

# === 6. Simpan hasil ===
df.to_csv("candidate_slang_normalized.csv", index=False, encoding="utf-8")

# === 7. Simpan kata yang gagal ===
unrecognized = df[df["correct_word"].isna()]["slang_word"].unique().tolist()
with open("unrecognized_words.txt", "w", encoding="utf-8") as f:
    for w in unrecognized:
        f.write(w + "\n")

print("✅ Normalisasi selesai!")
print("📂 Hasil: candidate_slang_normalized.csv")
print("📂 Kata gagal dikenali: unrecognized_words.txt")
