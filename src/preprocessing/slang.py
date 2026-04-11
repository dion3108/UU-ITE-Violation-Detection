import pandas as pd
import difflib

# === 1. Load data slang ===
df = pd.read_csv("Data/candidate_slang.csv")

# === 2. Load kamus KBBI ===
with open("Data/kbbi.txt", "r", encoding="utf-8") as f:
    kbbi_words = [line.strip().lower() for line in f if line.strip()]
kbbi_set = set(kbbi_words)

# === 3. Buat mapping manual (isi sesuai kebutuhan) ===
manual_mapping = {
    "anjirrrrrr": "anjir",
    "apankabr": "apa kabar",
    "yamg": "yang",
    "meluk": "peluk",
    "wleeee": "weleh",
    "kerreeeeenn": "keren",
    "gorangan": "gorengan",
    "beliu": "beliau",
    "pemirikiran": "pemikiran",
    "memerhati": "memerhatikan",
    "ditutup2i": "ditutupi",
    "kaburrr": "kabur",
    "kageeett": "kaget",
    "cowo": "cowok",
    "nganjing": "anjing",
    # tambahkan mapping lain sesuai kebutuhan
}

# === 4. Fungsi koreksi hybrid ===
def correct_word(word):
    w = str(word).lower()

    # 1) cek mapping manual
    if w in manual_mapping:
        return manual_mapping[w]

    # 2) cek apakah kata ada di kamus langsung
    if w in kbbi_set:
        return w

    # 3) approximate match dengan filter panjang ±2 huruf
    subset = [kw for kw in kbbi_words if abs(len(kw) - len(w)) <= 2 and kw[0] == w[0]]
    candidates = difflib.get_close_matches(w, subset, n=1, cutoff=0.8)
    return candidates[0] if candidates else None

# === 5. Terapkan ke dataframe ===
df["correct_word"] = df["slang_word"].apply(correct_word)

# === 6. Simpan hasil ===
df.to_csv("candidate_slang_normalized.csv", index=False, encoding="utf-8")

print("Proses selesai! Hasil tersimpan di candidate_slang_normalized.csv")
