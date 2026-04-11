import pandas as pd
import re

# =====================
# Load Dataset
# =====================
df = pd.read_csv("Data/preprocessed_dataset.csv")

# =====================
# Ekstraksi Special Tokens
# =====================
all_text = " ".join(df["normal_teks"].astype(str).tolist())

# Cari token khusus (emoji_, mention_, link)
special_tokens = set(re.findall(r'\b(?:emoji_\w+|mention_\w+|link\d*)\b', all_text))

# Urutkan biar konsisten
special_tokens_list = sorted(list(special_tokens))

print(f"✅ Ditemukan {len(special_tokens_list)} special tokens")

# =====================
# Simpan ke TXT
# =====================
with open("special_tokens.txt", "w", encoding="utf-8") as f:
    for token in special_tokens_list:
        f.write(token + "\n")

print("✅ Special tokens disimpan ke special_tokens.txt")
