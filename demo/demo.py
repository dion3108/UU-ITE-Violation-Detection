import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
import pickle
import re
import emoji
import os
from unidecode import unidecode
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi UU ITE", page_icon="⚖️", layout="centered")

# --- JUDUL ---
st.title("⚖️ Demo Deteksi Pelanggaran UU ITE")
st.markdown("Aplikasi klasifikasi komentar multi-model: **Hybrid CNN-BiLSTM** dan **IndoBERT**.")

# --- KONSTANTA & LABEL ---
MAX_LEN_KERAS = 256
MODEL_NAME_BERT = "indobenchmark/indobert-base-p2"
LABELS_MAP = {
    0: "Netral",
    1: "Judi Online",
    2: "Berita Hoaks",
    3: "Pencemaran Nama Baik",
    4: "Ujaran Kebencian (SARA)",
    5: "Pornografi"
}

# --- 1. LOAD DATA PENDUKUNG (MENTION & SLANG) ---
@st.cache_data
def load_auxiliary_data():
    data = {}
    
    # A. Load List Judi Mentions
    # Note: Menggunakan path relative naik satu folder (..\Data) sesuai struktur kamu
    judi_path = os.path.join('..', 'Data', 'judi_mentions.csv')
    try:
        if os.path.exists(judi_path):
            judi_df = pd.read_csv(judi_path)
            data['judi_mentions'] = set(judi_df.iloc[:, 0].astype(str).str.lower())
        else:
            # Fallback jika file tidak ketemu di folder Data, cari di folder yang sama dengan app
            if os.path.exists('judi_mentions.csv'):
                 judi_df = pd.read_csv('judi_mentions.csv')
                 data['judi_mentions'] = set(judi_df.iloc[:, 0].astype(str).str.lower())
            else:
                st.warning(f"⚠️ File '{judi_path}' tidak ditemukan. Deteksi mention judi mungkin kurang akurat.")
                data['judi_mentions'] = set()
    except Exception as e:
        st.warning(f"⚠️ Gagal load judi_mentions: {e}")
        data['judi_mentions'] = set()

    # B. Load Kamus Slang
    slang_path = os.path.join('..', 'Data', 'kamus-slang.csv')
    try:
        if os.path.exists(slang_path):
            slang_df = pd.read_csv(slang_path)
            data['slang_dict'] = dict(zip(slang_df.iloc[:, 0].astype(str), slang_df.iloc[:, 1].astype(str)))
        else:
            # Fallback
            if os.path.exists('kamus-slang.csv'):
                slang_df = pd.read_csv('kamus-slang.csv')
                data['slang_dict'] = dict(zip(slang_df.iloc[:, 0].astype(str), slang_df.iloc[:, 1].astype(str)))
            else:
                st.warning(f"⚠️ File '{slang_path}' tidak ditemukan. Normalisasi slang dilewati.")
                data['slang_dict'] = {}
    except Exception as e:
        st.warning(f"⚠️ Gagal load kamus-slang: {e}")
        data['slang_dict'] = {}
        
    return data

aux_data = load_auxiliary_data()

# --- 2. FUNGSI PREPROCESSING UTAMA ---
def full_preprocessing(text, aux_data):
    if not isinstance(text, str): return ""
    
    # 1. Case Folding
    text = text.lower()
    
    # 2. Demojize
    def demojize_text(t):
        for em in emoji.emoji_list(t):
            t = t.replace(em['emoji'], f" {em['emoji']} ")
        t = re.sub(r'\s+', ' ', t).strip()
        demojized = emoji.demojize(t)
        demojized = re.sub(r':([^:]+):', r'emoji_\1', demojized)
        return demojized
    
    text = demojize_text(text)
    
    # 3. Normalize special char to ASCII
    text = unidecode(text)
    
    # 4. Split Compound Words
    text = re.sub(r'([a-zA-Z]+)(\d+)([a-zA-Z]+)', r'\1 \2 \3', text)
    
    # 5. Handle URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_pattern, 'link0', text)
    
    # 6. Handle Mentions
    judi_mentions = aux_data.get('judi_mentions', set())
    patterns = [r'\d{3}$', r'hoki', r'parlay']
    judi_patterns = re.compile('|'.join(patterns))

    def is_judi_mention(username):
        return (username in judi_mentions) or (judi_patterns.search(username) is not None)

    def replacer(match):
        username = match.group(1).lower()
        if is_judi_mention(username):
            return f'mention_{username}'
        else:
            return 'mention_user'
            
    text = re.sub(r'@(\w+)', replacer, text)
    
    # 7. Remove Hashtags
    text = re.sub(r'#\w+', '', text)
    
    # 8. Clean Punctuation (IndoBERT style)
    text = re.sub(r"(\w)([?!])", r"\1 \2", text)
    import string
    punct_to_remove = string.punctuation.replace("?", "").replace("!", "").replace("-", "")
    text = text.translate(str.maketrans("", "", punct_to_remove))
    
    # 9. Slang Normalization
    slang_mapping = aux_data.get('slang_dict', {})
    if slang_mapping:
        words = text.split()
        normalized = [slang_mapping.get(w, w) for w in words]
        text = ' '.join(normalized)
        
    return text.lower().strip()

# # --- 3. LOAD MODEL & RESOURCES versi jalan tapi cnn gagal ---
# @st.cache_resource
# def load_all_models():
#     resources = {}
    
#     # === A. LOAD KERAS RESOURCES ===
#     # Path menggunakan os.path.join agar aman di Windows
#     keras_folder = "Cnn-BiLSTM_Model"
#     try:
#         model_path = os.path.join(keras_folder, 'best_cnn_bilstm_trainable.h5')
#         tok_path = os.path.join(keras_folder, 'tokenizer_cnn.pickle')
#         le_path = os.path.join(keras_folder, 'label_encoder.pickle')

#         resources['model_keras'] = tf.keras.models.load_model(model_path)
#         with open(tok_path, 'rb') as f:
#             resources['tokenizer_keras'] = pickle.load(f)
#         with open(le_path, 'rb') as f:
#             resources['le'] = pickle.load(f)
#     except Exception as e:
#         st.error(f"⚠️ Gagal memuat aset Keras (CNN-BiLSTM): {e}")

#     # === B. LOAD INDOBERT RESOURCES ===
#     bert_folder = "Indobert_Model"
#     try:
#         # 1. Load Tokenizer
#         # Cek apakah folder tokenizer custom ada
#         local_tokenizer_path = os.path.join(bert_folder, "tokenizer")
#         if os.path.exists(local_tokenizer_path):
#             tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)
#         else:
#             st.warning("⚠️ Folder 'tokenizer' tidak ditemukan di dalam Indobert_Model. Menggunakan tokenizer default (bisa menyebabkan error shape mismatch).")
#             tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_BERT)
        
#         resources['tokenizer_bert'] = tokenizer
        
#         # 2. Load Arsitektur Model Kosong
#         model_bert = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_BERT, num_labels=6)
        
#         # 3. [PENTING] Resize Embeddings agar sesuai dengan Tokenizer yang sudah ditraining
#         # Ini memperbaiki error KeyError dan Shape Mismatch
#         model_bert.resize_token_embeddings(len(tokenizer))
        
#         # 4. Load Weights (State Dict)
#         weights_path = os.path.join(bert_folder, 'best_model.pt')
#         state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
#         model_bert.load_state_dict(state_dict)
#         model_bert.eval()
        
#         resources['model_bert'] = model_bert
        
#     except Exception as e:
#         st.error(f"⚠️ Gagal memuat aset IndoBERT: {e}")
#         st.write("Tips: Pastikan folder 'Indobert_Model' berisi 'best_model.pt' DAN folder 'tokenizer'.")

#     return resources

# --- 3. LOAD MODEL & RESOURCES ---
@st.cache_resource
def load_all_models():
    resources = {}
    
    # === A. LOAD KERAS RESOURCES (ULTIMATE PATCH) ===
    keras_folder = "Cnn-BiLSTM_Model"
    try:
        model_path = os.path.join(keras_folder, 'best_cnn_bilstm_trainable.h5')
        tok_path = os.path.join(keras_folder, 'tokenizer_cnn.pickle')
        le_path = os.path.join(keras_folder, 'label_encoder.pickle')

        # --- 1. FUNGSI PEMBERSIH CONFIG (HAPUS DTypePolicy) ---
        def clean_config(config):
            # Jika ada key 'dtype' yang isinya dictionary (fitur Keras 3),
            # ganti paksa jadi string 'float32' (standar Keras 2)
            if 'dtype' in config and isinstance(config['dtype'], dict):
                config['dtype'] = 'float32'
            return config

        # --- 2. CLASS PENAMBAL (CUSTOM LAYERS) ---
        
        # Fix InputLayer (Masalah batch_shape)
        class FixedInputLayer(tf.keras.layers.InputLayer):
            def __init__(self, batch_shape=None, **kwargs):
                if batch_shape is not None:
                    kwargs['batch_input_shape'] = batch_shape
                super(FixedInputLayer, self).__init__(**kwargs)

        # Fix Embedding, Conv1D, LSTM, Dense, dll (Masalah DTypePolicy)
        class FixedEmbedding(tf.keras.layers.Embedding):
            @classmethod
            def from_config(cls, config):
                return super().from_config(clean_config(config))

        class FixedConv1D(tf.keras.layers.Conv1D):
            @classmethod
            def from_config(cls, config):
                return super().from_config(clean_config(config))
        
        class FixedLSTM(tf.keras.layers.LSTM):
            @classmethod
            def from_config(cls, config):
                return super().from_config(clean_config(config))
        
        class FixedDense(tf.keras.layers.Dense):
            @classmethod
            def from_config(cls, config):
                return super().from_config(clean_config(config))
                
        class FixedDropout(tf.keras.layers.Dropout):
            @classmethod
            def from_config(cls, config):
                return super().from_config(clean_config(config))
        
        class FixedMaxPooling1D(tf.keras.layers.MaxPooling1D):
             @classmethod
             def from_config(cls, config):
                 return super().from_config(clean_config(config))
        
        # [BARU] Fix Flatten (Penyebab Error Terakhir)
        class FixedFlatten(tf.keras.layers.Flatten):
             @classmethod
             def from_config(cls, config):
                 return super().from_config(clean_config(config))

        # Fix Bidirectional (Agak tricky karena dia membungkus layer lain)
        class FixedBidirectional(tf.keras.layers.Bidirectional):
            @classmethod
            def from_config(cls, config):
                # Bersihkan config layer di dalam wrapper Bidirectional
                if 'layer' in config and 'config' in config['layer']:
                     config['layer']['config'] = clean_config(config['layer']['config'])
                return super().from_config(clean_config(config))

        # --- 3. LOAD MODEL DENGAN DAFTAR PENAMBAL ---
        custom_objects_list = {
            'InputLayer': FixedInputLayer,
            'Embedding': FixedEmbedding,
            'Conv1D': FixedConv1D,
            'LSTM': FixedLSTM,
            'Dense': FixedDense,
            'Dropout': FixedDropout,
            'MaxPooling1D': FixedMaxPooling1D,
            'Flatten': FixedFlatten,        # <--- Sudah ditambahkan
            'Bidirectional': FixedBidirectional
        }

        # Load tanpa compile agar lebih ringan
        resources['model_keras'] = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects_list,
            compile=False 
        )
        
        with open(tok_path, 'rb') as f:
            resources['tokenizer_keras'] = pickle.load(f)
        with open(le_path, 'rb') as f:
            resources['le'] = pickle.load(f)
            
    except Exception as e:
        st.error(f"⚠️ Gagal memuat aset Keras (CNN-BiLSTM): {e}")
        st.write("Jika masih error layer lain, hubungi saya untuk metode 'Rebuild Architecture'.")

    # === B. LOAD INDOBERT RESOURCES ===
    bert_folder = "Indobert_Model"
    try:
        local_tokenizer_path = os.path.join(bert_folder, "tokenizer")
        if os.path.exists(local_tokenizer_path):
            tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)
        else:
            st.warning("⚠️ Menggunakan tokenizer default IndoBERT.")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_BERT)
        
        resources['tokenizer_bert'] = tokenizer
        
        model_bert = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_BERT, num_labels=6)
        model_bert.resize_token_embeddings(len(tokenizer))
        
        weights_path = os.path.join(bert_folder, 'best_model.pt')
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        model_bert.load_state_dict(state_dict)
        model_bert.eval()
        
        resources['model_bert'] = model_bert
        
    except Exception as e:
        st.error(f"⚠️ Gagal memuat aset IndoBERT: {e}")

    return resources
# Load semua model di awal
data = load_all_models()

# --- 4. USER INTERFACE ---
st.sidebar.header("🎛️ Pengaturan")
model_choice = st.sidebar.radio("Pilih Model:", ["IndoBERT (Transformer)", "CNN-BiLSTM (Hybrid)"])

st.subheader("Analisis Komentar")
input_text = st.text_area("Masukkan teks komentar:", height=150, placeholder="Contoh: @bandar88 situs ini gacor banget parlay aman...")

if st.button("🔍 Deteksi Pelanggaran"):
    if not input_text.strip():
        st.warning("Mohon masukkan teks.")
    elif data is None:
        st.error("Gagal memuat semua model. Periksa terminal untuk detail error.")
    else:
        with st.spinner("Sedang memproses..."):
            
            # 1. Jalankan Preprocessing
            processed_text = full_preprocessing(input_text, aux_data)
            
            with st.expander("Lihat Hasil Preprocessing"):
                st.code(processed_text)

            # 2. Prediksi
            prob_array = None
            pred_idx = 0
            
            # --- SKENARIO INDOBERT ---
            if model_choice == "IndoBERT (Transformer)":
                # Safety Check: Pastikan model benar-benar ter-load
                if 'model_bert' not in data or 'tokenizer_bert' not in data:
                    st.error("❌ Model IndoBERT tidak siap digunakan (Gagal Load). Cek folder Indobert_Model.")
                    st.stop()

                inputs = data['tokenizer_bert'](
                    processed_text, return_tensors="pt", padding=True, truncation=True, max_length=256
                )
                with torch.no_grad():
                    outputs = data['model_bert'](**inputs)
                    probs = F.softmax(outputs.logits, dim=1)
                    pred_idx = torch.argmax(probs, dim=1).item()
                    prob_array = probs[0].numpy()
            
            # --- SKENARIO CNN-BiLSTM ---
            else:
                # Safety Check
                if 'model_keras' not in data:
                    st.error("❌ Model CNN-BiLSTM tidak siap digunakan (Gagal Load). Cek folder Cnn-BiLSTM_Model.")
                    st.stop()

                seq = data['tokenizer_keras'].texts_to_sequences([processed_text])
                padded = pad_sequences(seq, maxlen=MAX_LEN_KERAS, padding='post')
                preds = data['model_keras'].predict(padded)
                pred_idx = np.argmax(preds, axis=1)[0]
                prob_array = preds[0]

            # 3. Tampilkan Hasil
            confidence = prob_array[pred_idx]
            
            # Decode Label
            try:
                # Prioritas pakai label encoder yang tersimpan
                if 'le' in data:
                    label_name = data['le'].inverse_transform([pred_idx])[0]
                else:
                    label_name = LABELS_MAP.get(pred_idx, "Unknown")
            except:
                label_name = LABELS_MAP.get(pred_idx, "Unknown")

            st.divider()
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Logika warna (Asumsi 0 = Netral)
                # Sesuaikan string "Netral" dengan label asli dari datasetmu jika beda
                is_safe = (pred_idx == 0) or ("netral" in str(label_name).lower()) or ("tidak" in str(label_name).lower())
                
                if is_safe:
                    st.success(f"### ✅ {label_name}")
                else:
                    st.error(f"### 🚨 {label_name}")
                st.write(f"**Confidence:** `{confidence:.2%}`")

            with col2:
                # Grafik Probabilitas
                try:
                    if 'le' in data:
                        labels = data['le'].classes_
                    else:
                        labels = list(LABELS_MAP.values())
                except:
                    labels = [f"Class {i}" for i in range(len(prob_array))]
                
                chart_data = pd.DataFrame({
                    "Label": labels,
                    "Probability": prob_array
                }).set_index("Label")
                st.bar_chart(chart_data)