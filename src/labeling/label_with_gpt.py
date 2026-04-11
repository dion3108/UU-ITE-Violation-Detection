import pandas as pd
import json
import math
import os
import re
import time
import datetime
from g4f.client import Client
import matplotlib.pyplot as plt
from collections import Counter

# ========== KONFIGURASI ========== #
INPUT_FILE = "/kaggle/input/datauuiteusk0/cleaned_comments_filtered(cc1) copy.csv"
OUTPUT_FILE = "/kaggle/working/hasil_label_g4f_cleaned_comments_filtered(cc1).json"
OUTPUT_CSV_FILE = "/kaggle/working/hasil_label_g4f_cleaned_comments_filtered(cc1).csv"
CHECKPOINT_FILE = "/kaggle/working/checkpoint_g4f_cleaned_comments_filtered(cc1).txt"
ERROR_DIR = "/kaggle/working/error_batches(cc1)"
COLUMN_NAME = "T_komentar"
ID_COLUMN = "komentar_id"
BATCH_SIZE = 20

# ========== SETUP ========== #
client = Client()
os.makedirs(ERROR_DIR, exist_ok=True)

# ========== BACA DATA KOMENTAR ========== #
df = pd.read_csv(INPUT_FILE)
df = df.dropna(subset=[COLUMN_NAME, ID_COLUMN])
komentar_data = df[[ID_COLUMN, COLUMN_NAME]].astype(str).to_dict(orient="records")
total_batches = math.ceil(len(komentar_data) / BATCH_SIZE)

# ========== CEK CHECKPOINT ========== #
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r") as f:
        last_batch = int(f.read().strip())
else:
    last_batch = -1

# ========== MUAT HASIL SEBELUMNYA ========== #
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        all_results = json.load(f)
else:
    all_results = []

# ========== BUILD PROMPT (versi agresif, sudah diperbaiki bahasanya) ========== #
def build_prompt(batch):
    prompt = (
        "Anda adalah seorang ahli hukum digital Indonesia yang bertugas menganalisis dan melabeli komentar.\n"
        "Analisis setiap komentar berikut berdasarkan potensi pelanggaran UU ITE Indonesia.\n\n"
        "Gunakan HANYA label numerik berikut:\n"
        "5: Pornografi (kalimat vulgar, sugestif, seksual tersirat maupun eksplisit)\n"
        "2: Berita Hoaks (informasi palsu, bohong, klaim tidak berdasar, tanpa bukti)\n"
        "3: Pencemaran Nama Baik / Cyberbullying (tuduhan tidak benar, hinaan, sarkasme menyerang pribadi)\n"
        "4: Ujaran Kebencian SARA (provokasi atau hinaan berdasarkan Suku, Agama, Ras, Antargolongan yang berpotensi memicu kebencian)\n"
        "1: Judi Daring (ajakan atau promosi, termasuk istilah tersirat atau tautan ke situs judi)\n"
        "0: Netral (tidak mengandung pelanggaran atau merupakan opini biasa)\n\n"
        "Berikan respons HANYA dalam format JSON Array yang valid.\n"
        "Setiap objek dalam array harus berisi:\n"
        "- 'Komentar': isi komentar\n"
        "- 'Label': angka 0–5 sesuai kategori di atas\n"
        "- 'Alasan': penjelasan singkat mengapa komentar diberi label tersebut\n\n"
        "Contoh format:\n"
        "[\n"
        "  {\"Komentar\": \"Teks komentar pertama...\", \"Label\": 3, \"Alasan\": \"Mengandung hinaan personal.\"},\n"
        "  {\"Komentar\": \"Teks komentar kedua...\", \"Label\": 0, \"Alasan\": \"Komentar bersifat umum.\"}\n"
        "]\n\n"
        "Berikut adalah daftar komentar yang harus dianalisis:\n"
    )
    for i, item in enumerate(batch, 1):
        prompt += f"{i}. {item[COLUMN_NAME]}\n"
    return prompt

# ========== PARSING RESPON GPT ========== #
def extract_json(text):
    try:
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        if cleaned.startswith("`") and cleaned.endswith("`"):
            cleaned = cleaned[1:-1]
        cleaned = cleaned.strip()

        match = re.search(r"\[\s*{.*?}\s*\]", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            return json.loads(cleaned)
    except Exception as e:
        raise ValueError(f"Gagal parsing JSON: {e}")

# ========== VALIDASI FORMAT ========== #
def is_valid_result(item):
    try:
        if not (isinstance(item, dict)
                and "Komentar" in item
                and "Label" in item
                and "Alasan" in item):
            return False

        label = item["Label"]
        if isinstance(label, float) and label.is_integer():
            item["Label"] = int(label)
            label = item["Label"]

        return isinstance(label, int) and 0 <= label <= 5
    except:
        return False

# ========== REQUEST GPT (DENGAN FALLBACK) ========== #
def get_response_with_fallback(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Kamu adalah pakar hukum digital yang memahami UU ITE Indonesia."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        if "RATE LIMIT" in str(e).upper():
            print("⚠️ GPT-4o limit tercapai, fallback ke gpt-3.5-turbo...")
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Kamu adalah pakar hukum digital yang memahami UU ITE Indonesia."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            except Exception as fallback_error:
                raise RuntimeError(f"Gagal fallback ke gpt-3.5-turbo: {fallback_error}")
        else:
            raise e

# ========== LOOP UTAMA ========== #
for i in range(last_batch + 1, total_batches):
    batch = komentar_data[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
    prompt = build_prompt(batch)

    start_time = datetime.datetime.now()

    try:
        content = get_response_with_fallback(prompt)
        result = extract_json(content)

        if result is None or not all(is_valid_result(item) for item in result):
            raise ValueError("❌ Respons bukan JSON array valid atau mengandung label tidak sah.")

        for j, item in enumerate(result):
            if j < len(batch):
                item["komentar_id"] = batch[j]["komentar_id"]

        all_results.extend(result)

        # Simpan ke JSON
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        # Simpan juga ke CSV
        pd.DataFrame(all_results).to_csv(OUTPUT_CSV_FILE, index=False, encoding="utf-8-sig")

        with open(CHECKPOINT_FILE, "w") as f:
            f.write(str(i))

        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"✅ Batch {i+1}/{total_batches} selesai & disimpan! ⏱️ {duration:.2f} detik")

    except Exception as e:
        error_file = os.path.join(ERROR_DIR, f"error_batch_{i+1}.txt")
        with open(error_file, "w", encoding="utf-8") as f:
            f.write(f"PROMPT:\n{prompt}\n\nERROR:\n{str(e)}\n\nRESP:\n{locals().get('content', 'No response')}")
        print(f"❌ Gagal batch {i+1}: {e} (disimpan di {error_file})")

    time.sleep(1.5)

# ========== VISUALISASI DISTRIBUSI LABEL ========== #
label_counts = Counter([entry["Label"] for entry in all_results if isinstance(entry, dict) and "Label" in entry])

print("\n📊 Distribusi Label:")
for label, count in sorted(label_counts.items()):
    print(f"Label {label}: {count} komentar")

plt.figure(figsize=(8, 5))
plt.bar(label_counts.keys(), label_counts.values(), color='skyblue')
plt.title("Distribusi Kategori Label UU ITE")
plt.xlabel("Label")
plt.ylabel("Jumlah Komentar")
plt.xticks([0, 1, 2, 3, 4, 5])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("/kaggle/working/distribusi_label.png")
plt.show()
