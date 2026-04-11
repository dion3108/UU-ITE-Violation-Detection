import pandas as pd
import re
import unicodedata
import emoji

# =======================================
# 1. Load dataset
# =======================================
df = pd.read_csv("../Data/datasethoaks.csv")

# Pastikan kolom komentar tersedia
if "Komentar" not in df.columns:
    possible = [c for c in df.columns if "omen" in c.lower()]
    if possible:
        df.rename(columns={possible[0]: "Komentar"}, inplace=True)

# =======================================
# 2. Tambahkan kolom indeks (H-1, H-2, ...)
# =======================================
df.insert(0, "Indeks", [f"H-{i+1}" for i in range(len(df))])

# =======================================
# 3. Case folding (huruf kecil)
# =======================================
df["Komentar_bersih"] = df["Komentar"].astype(str).str.lower()

# =======================================
# 4. Demojize emojis (ubah ke token emoji_custom)
# =======================================
def demojize_text(text):
    """
    Mengubah emoji menjadi token custom, misalnya:
    😀 -> emoji_grinning_face
    😂 -> emoji_face_with_tears_of_joy
    ❤️ -> emoji_red_heart
    """
    if pd.isna(text):
        return ""
    text = str(text)

    # Tambahkan spasi di sekitar emoji
    for em in emoji.emoji_list(text):
        text = text.replace(em['emoji'], f" {em['emoji']} ")

    # Hilangkan spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()

    # Ubah emoji jadi deskripsi teks (mis. :grinning_face:)
    demojized = emoji.demojize(text)

    # Ubah :grinning_face: -> emoji_grinning_face
    demojized = re.sub(r':([^:]+):', r'emoji_\1', demojized)

    # Ganti tanda '-' di token menjadi '_'
    demojized = re.sub(r'emoji_([^\s]+)', lambda m: m.group(0).replace('-', '_'), demojized)

    return demojized

df["Komentar_bersih"] = df["Komentar_bersih"].apply(demojize_text)

# =======================================
# 5. Normalisasi karakter khusus (ke ASCII)
# =======================================
def normalize_to_ascii(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")

df["Komentar_bersih"] = df["Komentar_bersih"].apply(normalize_to_ascii)

# =======================================
# 6. Hapus URL
# =======================================
def replace_urls(text):
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    return re.sub(url_pattern, "link0", text)

df["Komentar_bersih"] = df["Komentar_bersih"].apply(replace_urls)

# =======================================
# 7. Ubah mention akun menjadi token "mention_user"
# =======================================
def replace_mentions(text):
    return re.sub(r"@\w+", "mention_user", text)

df["Komentar_bersih"] = df["Komentar_bersih"].apply(replace_mentions)

# =======================================
# 8. Hapus hashtag
# =======================================
df["Komentar_bersih"] = df["Komentar_bersih"].apply(lambda x: re.sub(r"#\w+", "", x))

# =======================================
# 9. Hapus tanda baca dan spasi berlebih
# =======================================
df["Komentar_bersih"] = df["Komentar_bersih"].apply(lambda x: re.sub(r"[^\w\s]", "", x))
df["Komentar_bersih"] = df["Komentar_bersih"].apply(lambda x: re.sub(r"\s+", " ", x).strip())

# =======================================
# 10. Buat kolom Label dari Label_Validator
# =======================================
if "Label_Validator" in df.columns:
    df["Label"] = df["Label_Validator"]
else:
    df["Label_Validator"] = ""
    df["Label"] = ""

# =======================================
# 11. Normalisasi slang (dari Komentar_bersih)
# =======================================
try:
    slang_dict = pd.read_csv("../Data/kamus-slang.csv")
    slang_mapping = dict(zip(slang_dict.iloc[:, 0].astype(str), slang_dict.iloc[:, 1].astype(str)))
    print(f"Loaded slang dictionary with {len(slang_mapping)} entries")
except Exception as e:
    slang_mapping = {}
    print(f"Warning: could not load slang dictionary: {e}")

def normalize_slang_text(text):
    if pd.isna(text):
        return ""
    words = text.split()
    normalized = [slang_mapping.get(w, w) for w in words]
    return " ".join(normalized)

df["normal_teks"] = df["Komentar_bersih"].apply(normalize_slang_text)

# =======================================
# 12. Simpan hasil akhir ke CSV
# =======================================
cols_to_save = ["Indeks", "Komentar", "Komentar_bersih", "normal_teks", "Label", "Label_Validator"]
df.to_csv("datasethoaks_bersih.csv", index=False, columns=cols_to_save)

print("✅ Preprocessing selesai! File disimpan sebagai datasethoaks_bersih.csv")
print(df[cols_to_save].head())
