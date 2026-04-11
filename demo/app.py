# import streamlit as st
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import torch
# import torch.nn.functional as F
# import pickle
# import re
# import emoji
# import os
# import json
# from unidecode import unidecode
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, Flatten
# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing.text import tokenizer_from_json
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# # --- KONFIGURASI HALAMAN ---
# st.set_page_config(page_title="Deteksi UU ITE", page_icon="⚖️", layout="centered")
# st.title("⚖️ Demo Deteksi Pelanggaran UU ITE")
# st.markdown("Aplikasi klasifikasi komentar multi-model: **Hybrid CNN-BiLSTM** dan **IndoBERT**.")

# # --- KONSTANTA ---
# MAX_LEN_KERAS = 256
# MODEL_NAME_BERT = "indobenchmark/indobert-base-p2"
# LABELS_MAP = {
#     0: "Netral",
#     1: "Judi Online",
#     2: "Berita Hoaks",
#     3: "Pencemaran Nama Baik",
#     4: "Ujaran Kebencian (SARA)",
#     5: "Pornografi"
# }

# # --- 1. LOAD DATA PENDUKUNG ---
# @st.cache_data
# def load_auxiliary_data():
#     data = {}
#     # Path folder Data (Naik satu level dari folder Demo)
#     judi_path = os.path.join('..', 'Data', 'judi_mentions.csv')
#     slang_path = os.path.join('..', 'Data', 'kamus-slang.csv')
    
#     # A. Load Judi Mentions
#     try:
#         if os.path.exists(judi_path):
#             judi_df = pd.read_csv(judi_path)
#             data['judi_mentions'] = set(judi_df.iloc[:, 0].astype(str).str.lower())
#         else:
#             data['judi_mentions'] = set()
#     except: data['judi_mentions'] = set()

#     # B. Load Kamus Slang
#     try:
#         if os.path.exists(slang_path):
#             slang_df = pd.read_csv(slang_path)
#             data['slang_dict'] = dict(zip(slang_df.iloc[:, 0].astype(str), slang_df.iloc[:, 1].astype(str)))
#         else:
#             data['slang_dict'] = {}
#     except: data['slang_dict'] = {}
        
#     return data

# aux_data = load_auxiliary_data()

# # --- 2. FUNGSI PREPROCESSING ---
# def full_preprocessing(text, aux_data):
#     if not isinstance(text, str): return ""
#     text = text.lower()
    
#     # Demojize
#     for em in emoji.emoji_list(text):
#         text = text.replace(em['emoji'], f" {em['emoji']} ")
#     text = re.sub(r'\s+', ' ', text).strip()
#     text = emoji.demojize(text)
#     text = re.sub(r':([^:]+):', r'emoji_\1', text)
    
#     text = unidecode(text)
#     text = re.sub(r'([a-zA-Z]+)(\d+)([a-zA-Z]+)', r'\1 \2 \3', text)
#     text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link0', text)
    
#     # Mention Handling
#     judi_mentions = aux_data.get('judi_mentions', set())
#     judi_patterns = re.compile(r'\d{3}$|hoki|parlay')
#     def replacer(match):
#         u = match.group(1).lower()
#         if (u in judi_mentions) or (judi_patterns.search(u) is not None):
#             return f'mention_{u}'
#         return 'mention_user'
#     text = re.sub(r'@(\w+)', replacer, text)
    
#     text = re.sub(r'#\w+', '', text)
    
#     # Punctuation (IndoBERT style)
#     text = re.sub(r"(\w)([?!])", r"\1 \2", text)
#     import string
#     punct = string.punctuation.replace("?", "").replace("!", "").replace("-", "")
#     text = text.translate(str.maketrans("", "", punct))
    
#     # Slang Normalization
#     slang = aux_data.get('slang_dict', {})
#     if slang:
#         text = ' '.join([slang.get(w, w) for w in text.split()])
        
#     return text.lower().strip()

# # --- 3. REBUILD ARSITEKTUR MODEL (Sesuai Konfigurasi User: Kernel 7) ---
# def build_cnn_bilstm_architecture(vocab_size):
#     embed_dim = 300
#     max_len = 256
    
#     inp = Input(shape=(max_len,), name="input_ids")
    
#     # Embedding
#     emb = Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len, trainable=True)(inp)
#     x = Dropout(0.2)(emb)

#     # Block CNN 1 (Kernel Size 7)
#     x = Conv1D(filters=64, kernel_size=7, activation="relu", padding="valid")(x)
#     x = MaxPooling1D(pool_size=3)(x)
#     x = Dropout(0.3)(x)

#     # Block CNN 2 (Kernel Size 7)
#     x = Conv1D(filters=64, kernel_size=7, activation="relu", padding="same")(x)
#     x = MaxPooling1D(pool_size=3)(x)
#     x = Dropout(0.3)(x)

#     # BiLSTM
#     x = Bidirectional(LSTM(64, return_sequences=True))(x)
#     x = Dropout(0.3)(x)
#     x = Flatten()(x)

#     # Dense
#     x = Dense(128, activation="relu")(x)
#     x = Dropout(0.5)(x)
#     out = Dense(6, activation="softmax")(x)

#     model = Model(inputs=inp, outputs=out)
#     return model

# # --- 4. LOAD MODEL & RESOURCES ---
# @st.cache_resource
# def load_all_models():
#     resources = {}
#     keras_folder = "Cnn-BiLSTM_Model"
#     bert_folder = "Indobert_Model"
    
#     # === A. LOAD KERAS ===
#     try:
#         # 1. Load Tokenizer JSON (Dengan Fix 'str' attribute)
#         tok_path = os.path.join(keras_folder, 'tokenizer_cnnBiLSTM.json')
        
#         # Fallback nama file jika user lupa rename
#         if not os.path.exists(tok_path):
#              tok_path = os.path.join(keras_folder, 'tokenizer_cnn.json')
             
#         if not os.path.exists(tok_path):
#              st.error(f"❌ File tokenizer JSON tidak ditemukan di folder {keras_folder}")
#              return None
             
#         with open(tok_path, 'r', encoding='utf-8') as f:
#             raw_json = json.load(f)
#             # LOGIKA FIX: Cek apakah hasil load berupa string atau dict
#             if isinstance(raw_json, str):
#                 # Jika string, langsung pakai
#                 tokenizer_keras = tokenizer_from_json(raw_json)
#             else:
#                 # Jika dict, ubah jadi string dulu
#                 tokenizer_keras = tokenizer_from_json(json.dumps(raw_json))
            
#         resources['tokenizer_keras'] = tokenizer_keras
        
#         # 2. Rebuild Architecture (Kernel 7) & Load Weights
#         vocab_size = len(tokenizer_keras.word_index) + 1
#         model_keras = build_cnn_bilstm_architecture(vocab_size)
        
#         weights_path = os.path.join(keras_folder, 'best_cnn_bilstm_trainable.h5')
#         model_keras.load_weights(weights_path)
#         resources['model_keras'] = model_keras
        
#         # 3. Load Label Encoder
#         le_path = os.path.join(keras_folder, 'label_encoder.pickle')
#         with open(le_path, 'rb') as f:
#             resources['le'] = pickle.load(f)
            
#     except Exception as e:
#         st.error(f"⚠️ Gagal memuat CNN-BiLSTM: {e}")
#         st.write("Tips: Pastikan file tokenizer adalah JSON dan kernel_size sesuai.")

#     # === B. LOAD INDOBERT ===
#     try:
#         local_tok = os.path.join(bert_folder, "tokenizer")
#         if os.path.exists(local_tok):
#             tokenizer = AutoTokenizer.from_pretrained(local_tok)
#         else:
#             tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_BERT)
#         resources['tokenizer_bert'] = tokenizer
        
#         model_bert = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_BERT, num_labels=6)
#         model_bert.resize_token_embeddings(len(tokenizer))
        
#         state_dict = torch.load(os.path.join(bert_folder, 'best_model.pt'), map_location=torch.device('cpu'))
#         model_bert.load_state_dict(state_dict)
#         model_bert.eval()
#         resources['model_bert'] = model_bert
        
#     except Exception as e:
#         st.error(f"⚠️ Gagal memuat IndoBERT: {e}")

#     return resources

# data = load_all_models()

# # --- 5. USER INTERFACE ---
# st.sidebar.header("🎛️ Pengaturan")
# model_choice = st.sidebar.radio("Pilih Model:", ["IndoBERT (Transformer)", "CNN-BiLSTM (Hybrid)"])

# st.subheader("Analisis Komentar")
# input_text = st.text_area("Masukkan teks komentar:", height=150, placeholder="Contoh: @bandar88 situs ini gacor banget parlay aman...")

# if st.button("🔍 Deteksi Pelanggaran"):
#     if not input_text.strip():
#         st.warning("Mohon masukkan teks.")
#     elif data is None:
#         st.error("Gagal init model.")
#     else:
#         with st.spinner("Sedang memproses..."):
#             # 1. Preprocessing
#             processed_text = full_preprocessing(input_text, aux_data)
            
#             with st.expander("Lihat Hasil Preprocessing"):
#                 st.code(processed_text)

#             prob_array = None
#             pred_idx = 0
#             chart_labels = []

#             # 2. Prediksi & Penentuan Label
#             if model_choice == "IndoBERT (Transformer)":
#                 # === LOGIKA INDOBERT ===
#                 if 'model_bert' not in data: st.error("Model IndoBERT Error"); st.stop()
                
#                 inputs = data['tokenizer_bert'](processed_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
#                 with torch.no_grad():
#                     outputs = data['model_bert'](**inputs)
#                     probs = F.softmax(outputs.logits, dim=1)
#                     pred_idx = torch.argmax(probs, dim=1).item()
#                     prob_array = probs[0].numpy()
                
#                 # IndoBERT dilatih dengan angka, jadi kita MAPPING MANUAL ke Teks
#                 # Menggunakan LABELS_MAP (0=Netral, 1=Judi, dst...)
#                 label_name = LABELS_MAP.get(pred_idx, f"Label {pred_idx}")
                
#                 # Label untuk Chart (Ambil dari LABELS_MAP agar urut 0-5)
#                 chart_labels = [LABELS_MAP.get(i, str(i)) for i in range(len(prob_array))]

#             else:
#                 # === LOGIKA KERAS CNN-BiLSTM ===
#                 if 'model_keras' not in data: st.error("Model CNN-BiLSTM Error"); st.stop()
                
#                 seq = data['tokenizer_keras'].texts_to_sequences([processed_text])
#                 padded = pad_sequences(seq, maxlen=MAX_LEN_KERAS, padding='post')
#                 preds = data['model_keras'].predict(padded)
#                 pred_idx = np.argmax(preds, axis=1)[0]
#                 prob_array = preds[0]
                
#                 # Coba decode pakai LabelEncoder bawaan Keras
#                 try:
#                     # Ambil hasil decode (bisa teks, bisa angka)
#                     raw_label = data['le'].inverse_transform([pred_idx])[0]
                    
#                     # Jika hasilnya masih angka (misal '3' atau 3), terjemahkan pakai MAP
#                     if str(raw_label).isnumeric():
#                         label_name = LABELS_MAP.get(int(raw_label), str(raw_label))
#                     else:
#                         label_name = str(raw_label) # Sudah teks (misal "Judi Online")
                        
#                     # Siapkan label untuk chart
#                     raw_classes = data['le'].classes_
#                     chart_labels = []
#                     for c in raw_classes:
#                         if str(c).isnumeric():
#                             chart_labels.append(LABELS_MAP.get(int(c), str(c)))
#                         else:
#                             chart_labels.append(str(c))
                            
#                 except:
#                     # Fallback jika LabelEncoder error
#                     label_name = LABELS_MAP.get(pred_idx, "Unknown")
#                     chart_labels = list(LABELS_MAP.values())

#             # 3. Tampilkan Hasil Akhir
#             st.divider()
#             col1, col2 = st.columns([2, 1])
            
#             with col1:
#                 # Logika Warna (Hijau jika Netral/Tidak Melanggar)
#                 # Kita cek apakah label mengandung kata 'Netral' atau index-nya 0
#                 is_netral = (pred_idx == 0) or ("netral" in label_name.lower()) or ("tidak" in label_name.lower())
                
#                 if is_netral:
#                     st.success(f"### ✅ {label_name}")
#                 else:
#                     st.error(f"### 🚨 {label_name}")
                
#                 st.write(f"**Confidence:** `{prob_array[pred_idx]:.2%}`")
            
#             with col2:
#                 # Buat dataframe untuk chart
#                 # Pastikan panjang label sama dengan panjang probabilitas
#                 if len(chart_labels) != len(prob_array):
#                     # Fallback jika panjang beda
#                     chart_labels = [f"Class {i}" for i in range(len(prob_array))]

#                 df_chart = pd.DataFrame({"Label": chart_labels, "Prob": prob_array})
#                 df_chart = df_chart.set_index("Label")
#                 st.bar_chart(df_chart)



# #versi 2 yang jalank ada pasal

# import streamlit as st
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import torch
# import torch.nn.functional as F
# import pickle
# import re
# import emoji
# import os
# import json
# from unidecode import unidecode
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, Flatten
# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing.text import tokenizer_from_json
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# # --- KONFIGURASI HALAMAN ---
# st.set_page_config(page_title="Deteksi UU ITE", page_icon="⚖️", layout="centered")
# st.title("⚖️ Demo Deteksi Pelanggaran UU ITE")
# st.markdown("Aplikasi klasifikasi komentar multi-model: **Hybrid CNN-BiLSTM** dan **IndoBERT**.")

# # --- KONSTANTA & DATABASE PASAL ---
# MAX_LEN_KERAS = 256
# MODEL_NAME_BERT = "indobenchmark/indobert-base-p2"

# LABELS_MAP = {
#     0: "Netral",
#     1: "Judi Online",
#     2: "Berita Hoaks",
#     3: "Pencemaran Nama Baik",
#     4: "Ujaran Kebencian (SARA)",
#     5: "Pornografi"
# }

# # Database Pasal UU ITE (Berdasarkan UU No. 19 Tahun 2016)
# UU_ITE_INFO = {
#     0: {
#         "pasal": "-",
#         "isi": "Tidak ditemukan unsur pelanggaran UU ITE pada teks ini.",
#         "hukuman": "-"
#     },
#     1: { # Judi Online
#         "pasal": "Pasal 27 ayat (2) UU ITE",
#         "isi": '"Setiap Orang dengan sengaja dan tanpa hak mendistribusikan, mentransmisikan, dan/atau membuat dapat diaksesnya Informasi Elektronik dan/atau Dokumen Elektronik yang memiliki muatan perjudian."',
#         "hukuman": "Pidana penjara paling lama 6 (enam) tahun dan/atau denda paling banyak Rp1.000.000.000,00 (satu miliar rupiah)."
#     },
#     2: { # Berita Hoaks (Konteks Konsumen/Umum)
#         "pasal": "Pasal 28 ayat (1) UU ITE",
#         "isi": '"Setiap Orang dengan sengaja dan tanpa hak menyebarkan berita bohong dan menyesatkan yang mengakibatkan kerugian konsumen dalam Transaksi Elektronik."',
#         "hukuman": "Pidana penjara paling lama 6 (enam) tahun dan/atau denda paling banyak Rp1.000.000.000,00 (satu miliar rupiah)."
#     },
#     3: { # Pencemaran Nama Baik
#         "pasal": "Pasal 27 ayat (3) UU ITE",
#         "isi": '"Setiap Orang dengan sengaja dan tanpa hak mendistribusikan... Informasi Elektronik... yang memiliki muatan penghinaan dan/atau pencemaran nama baik."',
#         "hukuman": "Pidana penjara paling lama 4 (empat) tahun dan/atau denda paling banyak Rp750.000.000,00 (tujuh ratus lima puluh juta rupiah)."
#     },
#     4: { # Ujaran Kebencian (SARA)
#         "pasal": "Pasal 28 ayat (2) UU ITE",
#         "isi": '"Setiap Orang dengan sengaja dan tanpa hak menyebarkan informasi yang ditujukan untuk menimbulkan rasa kebencian atau permusuhan individu dan/atau kelompok masyarakat tertentu berdasarkan atas suku, agama, ras, dan antargolongan (SARA)."',
#         "hukuman": "Pidana penjara paling lama 6 (enam) tahun dan/atau denda paling banyak Rp1.000.000.000,00 (satu miliar rupiah)."
#     },
#     5: { # Pornografi
#         "pasal": "Pasal 27 ayat (1) UU ITE",
#         "isi": '"Setiap Orang dengan sengaja dan tanpa hak mendistribusikan... Informasi Elektronik... yang memiliki muatan yang melanggar kesusilaan."',
#         "hukuman": "Pidana penjara paling lama 6 (enam) tahun dan/atau denda paling banyak Rp1.000.000.000,00 (satu miliar rupiah)."
#     }
# }

# # --- 1. LOAD DATA PENDUKUNG ---
# @st.cache_data
# def load_auxiliary_data():
#     data = {}
#     judi_path = os.path.join('..', 'Data', 'judi_mentions.csv')
#     slang_path = os.path.join('..', 'Data', 'kamus-slang.csv')
    
#     try:
#         if os.path.exists(judi_path):
#             judi_df = pd.read_csv(judi_path)
#             data['judi_mentions'] = set(judi_df.iloc[:, 0].astype(str).str.lower())
#         else: data['judi_mentions'] = set()
#     except: data['judi_mentions'] = set()

#     try:
#         if os.path.exists(slang_path):
#             slang_df = pd.read_csv(slang_path)
#             data['slang_dict'] = dict(zip(slang_df.iloc[:, 0].astype(str), slang_df.iloc[:, 1].astype(str)))
#         else: data['slang_dict'] = {}
#     except: data['slang_dict'] = {}
        
#     return data

# aux_data = load_auxiliary_data()

# # --- 2. FUNGSI PREPROCESSING ---
# def full_preprocessing(text, aux_data):
#     if not isinstance(text, str): return ""
#     text = text.lower()
#     for em in emoji.emoji_list(text):
#         text = text.replace(em['emoji'], f" {em['emoji']} ")
#     text = re.sub(r'\s+', ' ', text).strip()
#     text = emoji.demojize(text)
#     text = re.sub(r':([^:]+):', r'emoji_\1', text)
#     text = unidecode(text)
#     text = re.sub(r'([a-zA-Z]+)(\d+)([a-zA-Z]+)', r'\1 \2 \3', text)
#     text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link0', text)
    
#     judi_mentions = aux_data.get('judi_mentions', set())
#     judi_patterns = re.compile(r'\d{3}$|hoki|parlay')
#     def replacer(match):
#         u = match.group(1).lower()
#         if (u in judi_mentions) or (judi_patterns.search(u) is not None):
#             return f'mention_{u}'
#         return 'mention_user'
#     text = re.sub(r'@(\w+)', replacer, text)
#     text = re.sub(r'#\w+', '', text)
#     text = re.sub(r"(\w)([?!])", r"\1 \2", text)
#     import string
#     punct = string.punctuation.replace("?", "").replace("!", "").replace("-", "")
#     text = text.translate(str.maketrans("", "", punct))
    
#     slang = aux_data.get('slang_dict', {})
#     if slang:
#         text = ' '.join([slang.get(w, w) for w in text.split()])
#     return text.lower().strip()

# # --- 3. REBUILD ARSITEKTUR MODEL (Kernel Size 7) ---
# def build_cnn_bilstm_architecture(vocab_size):
#     embed_dim = 300
#     max_len = 256
#     inp = Input(shape=(max_len,), name="input_ids")
#     emb = Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len, trainable=True)(inp)
#     x = Dropout(0.2)(emb)
#     # Kernel size 7 sesuai request
#     x = Conv1D(filters=64, kernel_size=7, activation="relu", padding="valid")(x)
#     x = MaxPooling1D(pool_size=3)(x)
#     x = Dropout(0.3)(x)
#     x = Conv1D(filters=64, kernel_size=7, activation="relu", padding="same")(x)
#     x = MaxPooling1D(pool_size=3)(x)
#     x = Dropout(0.3)(x)
#     # LSTM 64 sesuai perbaikan shape mismatch sebelumnya
#     x = Bidirectional(LSTM(64, return_sequences=True))(x)
#     x = Dropout(0.3)(x)
#     x = Flatten()(x)
#     x = Dense(128, activation="relu")(x)
#     x = Dropout(0.5)(x)
#     out = Dense(6, activation="softmax")(x)
#     model = Model(inputs=inp, outputs=out)
#     return model

# # --- 4. LOAD MODEL & RESOURCES ---
# @st.cache_resource
# def load_all_models():
#     resources = {}
#     keras_folder = "Cnn-BiLSTM_Model"
#     bert_folder = "Indobert_Model"
    
#     # === A. LOAD KERAS ===
#     try:
#         tok_path = os.path.join(keras_folder, 'tokenizer_cnnBiLSTM.json')
#         if not os.path.exists(tok_path): tok_path = os.path.join(keras_folder, 'tokenizer_cnn.json')
#         if not os.path.exists(tok_path):
#              st.error(f"❌ File tokenizer JSON tidak ditemukan di folder {keras_folder}")
#              return None
             
#         with open(tok_path, 'r', encoding='utf-8') as f:
#             raw_json = json.load(f)
#             if isinstance(raw_json, str): tokenizer_keras = tokenizer_from_json(raw_json)
#             else: tokenizer_keras = tokenizer_from_json(json.dumps(raw_json))
            
#         resources['tokenizer_keras'] = tokenizer_keras
#         vocab_size = len(tokenizer_keras.word_index) + 1
#         model_keras = build_cnn_bilstm_architecture(vocab_size)
        
#         weights_path = os.path.join(keras_folder, 'best_cnn_bilstm_trainable.h5')
#         model_keras.load_weights(weights_path)
#         resources['model_keras'] = model_keras
        
#         le_path = os.path.join(keras_folder, 'label_encoder.pickle')
#         with open(le_path, 'rb') as f: resources['le'] = pickle.load(f)
            
#     except Exception as e:
#         st.error(f"⚠️ Gagal memuat CNN-BiLSTM: {e}")

#     # === B. LOAD INDOBERT ===
#     try:
#         local_tok = os.path.join(bert_folder, "tokenizer")
#         if os.path.exists(local_tok): tokenizer = AutoTokenizer.from_pretrained(local_tok)
#         else: tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_BERT)
#         resources['tokenizer_bert'] = tokenizer
        
#         model_bert = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_BERT, num_labels=6)
#         model_bert.resize_token_embeddings(len(tokenizer))
        
#         state_dict = torch.load(os.path.join(bert_folder, 'best_model.pt'), map_location=torch.device('cpu'))
#         model_bert.load_state_dict(state_dict)
#         model_bert.eval()
#         resources['model_bert'] = model_bert
        
#     except Exception as e:
#         st.error(f"⚠️ Gagal memuat IndoBERT: {e}")

#     return resources

# data = load_all_models()

# # --- 5. USER INTERFACE ---
# st.sidebar.header("🎛️ Pengaturan")
# model_choice = st.sidebar.radio("Pilih Model:", ["IndoBERT (Transformer)", "CNN-BiLSTM (Hybrid)"])

# st.subheader("Analisis Komentar")
# input_text = st.text_area("Masukkan teks komentar:", height=150, placeholder="Contoh: @bandar88 situs ini gacor banget parlay aman...")

# if st.button("🔍 Deteksi Pelanggaran"):
#     if not input_text.strip():
#         st.warning("Mohon masukkan teks.")
#     elif data is None:
#         st.error("Gagal init model.")
#     else:
#         with st.spinner("Sedang memproses..."):
#             processed_text = full_preprocessing(input_text, aux_data)
            
#             with st.expander("Lihat Hasil Preprocessing"):
#                 st.code(processed_text)

#             prob_array = None
#             pred_idx = 0
            
#             # --- PREDIKSI ---
#             if model_choice == "IndoBERT (Transformer)":
#                 if 'model_bert' not in data: st.error("Model IndoBERT Error"); st.stop()
#                 inputs = data['tokenizer_bert'](processed_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
#                 with torch.no_grad():
#                     outputs = data['model_bert'](**inputs)
#                     probs = F.softmax(outputs.logits, dim=1)
#                     pred_idx = torch.argmax(probs, dim=1).item()
#                     prob_array = probs[0].numpy()
#             else:
#                 if 'model_keras' not in data: st.error("Model CNN-BiLSTM Error"); st.stop()
#                 seq = data['tokenizer_keras'].texts_to_sequences([processed_text])
#                 padded = pad_sequences(seq, maxlen=MAX_LEN_KERAS, padding='post')
#                 preds = data['model_keras'].predict(padded)
#                 pred_idx = np.argmax(preds, axis=1)[0]
#                 prob_array = preds[0]

#             # --- DECODE LABEL ---
#             # Kita paksa ambil label dari LABELS_MAP agar seragam & tidak muncul angka
#             label_name = LABELS_MAP.get(pred_idx, "Unknown")

#             # --- TAMPILKAN HASIL UTAMA ---
#             st.divider()
#             col1, col2 = st.columns([2, 1])
            
#             with col1:
#                 # Cek Netral
#                 if pred_idx == 0:
#                     st.success(f"### ✅ {label_name}")
#                 else:
#                     st.error(f"### 🚨 {label_name}")
                
#                 st.write(f"**Confidence Score:** `{prob_array[pred_idx]:.2%}`")
                
#                 # === BAGIAN BARU: INFO PASAL & HUKUMAN ===
#                 if pred_idx != 0: # Hanya tampilkan jika melanggar
#                     info_pasal = UU_ITE_INFO.get(pred_idx, {})
                    
#                     with st.expander("📜 Detail Pasal & Ancaman Hukuman", expanded=True):
#                         st.markdown(f"**Dasar Hukum:**")
#                         st.info(info_pasal.get('pasal', '-'))
                        
#                         st.markdown(f"**Bunyi Pasal:**")
#                         st.warning(f"_{info_pasal.get('isi', '-')}_")
                        
#                         st.markdown(f"**Ancaman Hukuman:**")
#                         st.error(info_pasal.get('hukuman', '-'))
#                 else:
#                     st.info("Komentar ini terdeteksi aman dan tidak mengandung unsur pelanggaran UU ITE.")

#             with col2:
#                 # Bar Chart
#                 try: labels = [LABELS_MAP.get(i, str(i)) for i in range(len(prob_array))]
#                 except: labels = [f"C{i}" for i in range(len(prob_array))]
                
#                 df_chart = pd.DataFrame({"Label": labels, "Prob": prob_array})
#                 df_chart = df_chart.set_index("Label")
#                 st.bar_chart(df_chart)


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
import json
from unidecode import unidecode
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi UU ITE1", page_icon="⚖️", layout="centered")
st.title("⚖️ Demo Deteksi Potensi Pelanggaran UU ITE")
st.markdown("Aplikasi klasifikasi komentar multi-model: **Hybrid CNN-BiLSTM** dan **IndoBERT**.")

# --- KONSTANTA & DATABASE PASAL ---
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

# Database Pasal UU ITE (Berdasarkan UU No. 19 Tahun 2016)
UU_ITE_INFO = {
    0: {
        "pasal": "-",
        "isi": "Tidak ditemukan unsur pelanggaran UU ITE pada teks ini.",
        "hukuman": "-"
    },
    1: { # Judi Online
        "pasal": "Pasal 27 ayat (2) UU ITE",
        "isi": '"Setiap Orang dengan sengaja dan tanpa hak mendistribusikan, mentransmisikan, dan/atau membuat dapat diaksesnya Informasi Elektronik dan/atau Dokumen Elektronik yang memiliki muatan perjudian."',
        "hukuman": "Pidana penjara paling lama 6 (enam) tahun dan/atau denda paling banyak Rp1.000.000.000,00 (satu miliar rupiah)."
    },
    2: { # Berita Hoaks (Konteks Konsumen/Umum)
        "pasal": "Pasal 28 ayat (1) UU ITE",
        "isi": '"Setiap Orang dengan sengaja dan tanpa hak menyebarkan berita bohong dan menyesatkan yang mengakibatkan kerugian konsumen dalam Transaksi Elektronik."',
        "hukuman": "Pidana penjara paling lama 6 (enam) tahun dan/atau denda paling banyak Rp1.000.000.000,00 (satu miliar rupiah)."
    },
    3: { # Pencemaran Nama Baik
        "pasal": "Pasal 27 ayat (3) UU ITE",
        "isi": '"Setiap Orang dengan sengaja dan tanpa hak mendistribusikan... Informasi Elektronik... yang memiliki muatan penghinaan dan/atau pencemaran nama baik."',
        "hukuman": "Pidana penjara paling lama 4 (empat) tahun dan/atau denda paling banyak Rp750.000.000,00 (tujuh ratus lima puluh juta rupiah)."
    },
    4: { # Ujaran Kebencian (SARA)
        "pasal": "Pasal 28 ayat (2) UU ITE",
        "isi": '"Setiap Orang dengan sengaja dan tanpa hak menyebarkan informasi yang ditujukan untuk menimbulkan rasa kebencian atau permusuhan individu dan/atau kelompok masyarakat tertentu berdasarkan atas suku, agama, ras, dan antargolongan (SARA)."',
        "hukuman": "Pidana penjara paling lama 6 (enam) tahun dan/atau denda paling banyak Rp1.000.000.000,00 (satu miliar rupiah)."
    },
    5: { # Pornografi
        "pasal": "Pasal 27 ayat (1) UU ITE",
        "isi": '"Setiap Orang dengan sengaja dan tanpa hak mendistribusikan... Informasi Elektronik... yang memiliki muatan yang melanggar kesusilaan."',
        "hukuman": "Pidana penjara paling lama 6 (enam) tahun dan/atau denda paling banyak Rp1.000.000.000,00 (satu miliar rupiah)."
    }
}

# --- 1. LOAD DATA PENDUKUNG ---
@st.cache_data
def load_auxiliary_data():
    data = {}
    judi_path = os.path.join('..', 'Data', 'judi_mentions.csv')
    slang_path = os.path.join('..', 'Data', 'kamus-slang.csv')
    
    try:
        if os.path.exists(judi_path):
            judi_df = pd.read_csv(judi_path)
            data['judi_mentions'] = set(judi_df.iloc[:, 0].astype(str).str.lower())
        else: data['judi_mentions'] = set()
    except: data['judi_mentions'] = set()

    try:
        if os.path.exists(slang_path):
            slang_df = pd.read_csv(slang_path)
            data['slang_dict'] = dict(zip(slang_df.iloc[:, 0].astype(str), slang_df.iloc[:, 1].astype(str)))
        else: data['slang_dict'] = {}
    except: data['slang_dict'] = {}
        
    return data

aux_data = load_auxiliary_data()

# --- 2. FUNGSI PREPROCESSING (Final Fix: URL & Invisible Char) ---
def full_preprocessing(text, aux_data):
    if not isinstance(text, str): return ""
    
    # 1. URL Handling (Updated Regex untuk support www.)
    # Pola: (http atau https atau www.) diikuti karakter URL
    url_pattern = r'(?:http[s]?://|www\.)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_pattern, 'link0', text)
    
    # 2. Case Folding
    text = text.lower()
    
    # 3. Demojize
    def demojize_text(t):
        for em in emoji.emoji_list(t):
            t = t.replace(em['emoji'], f" {em['emoji']} ")
        t = re.sub(r'\s+', ' ', t).strip()
        demojized = emoji.demojize(t)
        demojized = re.sub(r':([^:]+):', r'emoji_\1', demojized)
        return demojized
    
    text = demojize_text(text)
    
    # [FIX] Hapus Invisible Characters (ZWJ, Word Joiner, dll)
    text = re.sub(r'[\u200B-\u200D\u2060\uFEFF]', '', text)

    # 4. Normalize special char to ASCII
    text = unidecode(text)
    
    # 5. Split Compound Words
    text = re.sub(r'([a-zA-Z]+)(\d+)([a-zA-Z]+)', r'\1 \2 \3', text)
    
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
    
    # 8. Final Cleaning
    text = text.lower()
    
    # 9. Slang Normalization
    slang = aux_data.get('slang_dict', {})
    if slang:
        words = text.split()
        normalized = []
        for w in words:
            if w in slang:
                repl = slang[w]
                if pd.isna(repl) or not isinstance(repl, str):
                    normalized.append(w)
                else:
                    normalized.append(repl)
            else:
                normalized.append(w)
        text = ' '.join(normalized)
        
    return text.strip()

# --- 3. REBUILD ARSITEKTUR MODEL ---
def build_cnn_bilstm_architecture(vocab_size):
    embed_dim = 300
    max_len = 256
    inp = Input(shape=(max_len,), name="input_ids")
    emb = Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len, trainable=True)(inp)
    x = Dropout(0.2)(emb)
    # Kernel 7 & LSTM 64 (Sesuai perbaikan sebelumnya)
    x = Conv1D(filters=64, kernel_size=7, activation="relu", padding="valid")(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Dropout(0.3)(x)
    x = Conv1D(filters=64, kernel_size=7, activation="relu", padding="same")(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(6, activation="softmax")(x)
    model = Model(inputs=inp, outputs=out)
    return model

# --- 4. LOAD MODEL & RESOURCES ---
@st.cache_resource
def load_all_models():
    resources = {}
    keras_folder = "Cnn-BiLSTM_Model"
    bert_folder = "Indobert_Model"
    
    # === A. LOAD KERAS ===
    try:
        tok_path = os.path.join(keras_folder, 'tokenizer_cnnBiLSTM.json')
        if not os.path.exists(tok_path): tok_path = os.path.join(keras_folder, 'tokenizer_cnn.json')
        if not os.path.exists(tok_path):
             st.error(f"❌ File tokenizer JSON tidak ditemukan di folder {keras_folder}")
             return None
             
        with open(tok_path, 'r', encoding='utf-8') as f:
            raw_json = json.load(f)
            if isinstance(raw_json, str): tokenizer_keras = tokenizer_from_json(raw_json)
            else: tokenizer_keras = tokenizer_from_json(json.dumps(raw_json))
            
        resources['tokenizer_keras'] = tokenizer_keras
        vocab_size = len(tokenizer_keras.word_index) + 1
        model_keras = build_cnn_bilstm_architecture(vocab_size)
        
        weights_path = os.path.join(keras_folder, 'best_cnn_bilstm_trainable.h5')
        model_keras.load_weights(weights_path)
        resources['model_keras'] = model_keras
        
        le_path = os.path.join(keras_folder, 'label_encoder.pickle')
        with open(le_path, 'rb') as f: resources['le'] = pickle.load(f)
            
    except Exception as e:
        st.error(f"⚠️ Gagal memuat CNN-BiLSTM: {e}")

    # === B. LOAD INDOBERT ===
    try:
        local_tok = os.path.join(bert_folder, "tokenizer")
        if os.path.exists(local_tok): tokenizer = AutoTokenizer.from_pretrained(local_tok)
        else: tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_BERT)
        resources['tokenizer_bert'] = tokenizer
        
        model_bert = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_BERT, num_labels=6)
        model_bert.resize_token_embeddings(len(tokenizer))
        
        state_dict = torch.load(os.path.join(bert_folder, 'best_model.pt'), map_location=torch.device('cpu'))
        model_bert.load_state_dict(state_dict)
        model_bert.eval()
        resources['model_bert'] = model_bert
        
    except Exception as e:
        st.error(f"⚠️ Gagal memuat IndoBERT: {e}")

    return resources

data = load_all_models()

# --- 5. USER INTERFACE ---
st.sidebar.header("🎛️ Pengaturan")
model_choice = st.sidebar.radio("Pilih Model:", ["IndoBERT (Transformer)", "CNN-BiLSTM (Hybrid)"])

# ... (Kode atas tetap sama) ...

st.subheader("Analisis Komentar")
input_text = st.text_area("Masukkan teks komentar:", height=150, placeholder="Contoh: @bandar88 situs ini gacor banget parlay aman...")

if st.button("🔍 Deteksi Pelanggaran"):
    if not input_text.strip():
        st.warning("Mohon masukkan teks.")
    elif data is None:
        st.error("Gagal init model.")
    else:
        # 1. PREPROCESSING (Spinner Khusus Preprocessing)
        with st.spinner("🧹 Sedang membersihkan teks (Preprocessing)..."):
            processed_text = full_preprocessing(input_text, aux_data)
        
        # Tampilkan hasil preprocessing di luar spinner agar spinner hilang dulu
        with st.expander("Lihat Hasil Preprocessing"):
            st.code(processed_text)

        prob_array = None
        pred_idx = 0
        
        # 2. PREDIKSI (Spinner Spesifik per Model)
        if model_choice == "IndoBERT (Transformer)":
            # --- LOGIKA INDOBERT ---
            if 'model_bert' not in data: 
                st.error("Model IndoBERT Error")
                st.stop()
            
            # SPINNER KHUSUS INDOBERT (Disini kuncinya)
            with st.spinner("🤖 IndoBERT sedang menganalisis konteks kalimat..."):
                inputs = data['tokenizer_bert'](processed_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
                with torch.no_grad():
                    outputs = data['model_bert'](**inputs)
                    probs = F.softmax(outputs.logits, dim=1)
                    pred_idx = torch.argmax(probs, dim=1).item()
                    prob_array = probs[0].numpy()

        else:
            # --- LOGIKA CNN-BiLSTM ---
            if 'model_keras' not in data: 
                st.error("Model CNN-BiLSTM Error")
                st.stop()
            
            # SPINNER KHUSUS CNN-BILSTM
            with st.spinner("🧠 CNN-BiLSTM sedang mengolah pola kata..."):
                seq = data['tokenizer_keras'].texts_to_sequences([processed_text])
                padded = pad_sequences(seq, maxlen=MAX_LEN_KERAS, padding='post')
                preds = data['model_keras'].predict(padded)
                pred_idx = np.argmax(preds, axis=1)[0]
                prob_array = preds[0]

        # --- DECODE LABEL ---
        label_name = LABELS_MAP.get(pred_idx, "Unknown")

        # --- TAMPILKAN HASIL ---
        st.divider()
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Cek Aman/Tidak
            is_safe = (pred_idx == 0) or ("netral" in label_name.lower()) or ("tidak" in label_name.lower())
            
            if is_safe:
                st.success(f"### ✅ {label_name}")
            else:
                st.error(f"### 🚨 {label_name}")
            
            st.write(f"**Confidence:** `{prob_array[pred_idx]:.2%}`")
            
            # INFO PASAL
            if not is_safe:
                info_pasal = UU_ITE_INFO.get(pred_idx, {})
                with st.expander("📜 Detail Pasal & Hukuman", expanded=True):
                    st.markdown("**Dasar Hukum:**")
                    st.info(info_pasal.get('pasal', '-'))
                    st.markdown("**Bunyi Ringkas:**")
                    st.warning(f"_{info_pasal.get('isi', '-')}_")
                    st.markdown("**Ancaman:**")
                    st.error(info_pasal.get('hukuman', '-'))
            else:
                st.info("Tidak ditemukan unsur pidana UU ITE.")

        with col2:
            try: labels = [LABELS_MAP.get(i, str(i)) for i in range(len(prob_array))]
            except: labels = [f"C{i}" for i in range(len(prob_array))]
            
            df_chart = pd.DataFrame({"Label": labels, "Prob": prob_array})
            df_chart = df_chart.set_index("Label")
            st.bar_chart(df_chart)