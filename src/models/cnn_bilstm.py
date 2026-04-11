# -*- coding: utf ini yang dipakek tes stremalit-8 -*-
# CNN–BiLSTM (paper version, FastText trainable) dengan Sparse Categorical Crossentropy
# ================================================================

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.layers import (Input, Embedding, Conv1D, MaxPooling1D,
                                     Bidirectional, LSTM, Dense, Dropout, Flatten)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback

# =====================
# Load Dataset
# =====================
train_df = pd.read_csv("/kaggle/input/undersample-reduce-30k/train.csv")
val_df   = pd.read_csv("/kaggle/input/undersample-reduce-30k/val.csv")
test_df  = pd.read_csv("/kaggle/input/undersample-reduce-30k/test.csv")

# Bersihkan teks kosong
for name, df in zip(["Train", "Validation", "Test"], [train_df, val_df, test_df]):
    before = len(df)
    df["normal_teks"] = df["normal_teks"].astype(str)
    df.dropna(subset=["normal_teks"], inplace=True)
    df = df[df["normal_teks"].str.strip() != ""]
    after = len(df)
    print(f"{name}: {before} → {after} (dropped {before-after})")
    if name == "Train": train_df = df
    elif name == "Validation": val_df = df
    else: test_df = df

# =====================
# Encode label
# =====================
for d in (train_df, val_df, test_df):
    d["Label_Validator"] = d["Label_Validator"].astype(int)

le = LabelEncoder()
train_df["Label_Validator"] = le.fit_transform(train_df["Label_Validator"])
val_df["Label_Validator"]   = le.transform(val_df["Label_Validator"])
test_df["Label_Validator"]  = le.transform(test_df["Label_Validator"])
num_classes = len(le.classes_)

print("\n=== Dataset Info ===")
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"Classes: {num_classes} → {list(le.classes_)}")

# =====================
# Tokenization
# =====================
max_len = 256   # sesuai arsitektur paper
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(train_df["normal_teks"])
max_words = len(tokenizer.word_index) + 1

X_train = pad_sequences(tokenizer.texts_to_sequences(train_df["normal_teks"]), maxlen=max_len, padding="post")
X_val   = pad_sequences(tokenizer.texts_to_sequences(val_df["normal_teks"]),   maxlen=max_len, padding="post")
X_test  = pad_sequences(tokenizer.texts_to_sequences(test_df["normal_teks"]),  maxlen=max_len, padding="post")

# ✅ label integer
y_train = train_df["Label_Validator"].values
y_val   = val_df["Label_Validator"].values
y_test  = test_df["Label_Validator"].values

# =====================
# Load FastText .vec
# =====================
def load_fasttext_vec(file_path, word_index, embed_dim=300, vocab_size=None):
    embeddings_index = {}
    with open(file_path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            values = line.rstrip().split(" ")
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    
    print(f"Loaded {len(embeddings_index):,} vectors from {file_path}")
    
    if vocab_size is None:
        vocab_size = len(word_index) + 1
    
    embedding_matrix = np.zeros((vocab_size, embed_dim))
    hit = 0
    for word, i in word_index.items():
        if i >= vocab_size: continue
        vec = embeddings_index.get(word)
        if vec is not None:
            embedding_matrix[i] = vec
            hit += 1
    print(f"Embedding coverage: {hit}/{len(word_index)} ({hit/len(word_index)*100:.2f}%)")
    return embedding_matrix

embedding_matrix = load_fasttext_vec(
    "/kaggle/input/datatest/cc.id.300.vec",
    tokenizer.word_index,
    embed_dim=300,
    vocab_size=max_words
)

# =====================
# Build CNN–BiLSTM (trainable version)
# =====================
def build_cnn_bilstm_trainable(max_words,
                               max_len,
                               embedding_matrix,
                               num_classes=6,
                               embed_dim=300,
                               trainable=True,  # ✅ trainable FastText
                               conv_filters=64,
                               lstm_units=64,
                               dense_units=128,
                               drop_embed=0.2,
                               drop_conv1=0.3,
                               drop_conv2=0.3,
                               drop_bilstm=0.3,
                               drop_dense=0.5):
    inp = Input(shape=(max_len,), name="input_ids")

    emb = Embedding(
        input_dim=max_words,
        output_dim=embed_dim,
        weights=[embedding_matrix],
        input_length=max_len,
        trainable=trainable,
        name="embedding_fasttext_trainable"
    )(inp)

    x = Dropout(drop_embed)(emb)

    x = Conv1D(filters=conv_filters, kernel_size=7, activation="relu", padding="valid")(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Dropout(drop_conv1)(x)

    x = Conv1D(filters=conv_filters, kernel_size=7, activation="relu", padding="same")(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Dropout(drop_conv2)(x)

    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = Dropout(drop_bilstm)(x)
    x = Flatten()(x)

    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(drop_dense)(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",  # ✅ sparse loss
        metrics=["accuracy"]
    )
    return model

model = build_cnn_bilstm_trainable(
    max_words=max_words,
    max_len=max_len,
    embedding_matrix=embedding_matrix,
    num_classes=num_classes,
    embed_dim=300,
    trainable=True  # ✅ ubah ke trainable
)
model.summary()

# =====================
# Callback: EarlyStopping by F1
# =====================
class F1EarlyStopping(Callback):
    def __init__(self, validation_data, filepath, patience=5, log_file="training_log.csv"):
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.filepath = filepath
        self.best_f1 = -np.Inf
        self.patience = patience
        self.wait = 0
        self.log_file = log_file
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "val_loss", "val_acc", "val_f1"])

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = self.y_val
        f1 = f1_score(y_true, y_pred_classes, average="macro")
        val_loss = logs.get("val_loss")
        val_acc = logs.get("val_accuracy")

        print(f"\nEpoch {epoch+1}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_macro_f1={f1:.4f}")

        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f); writer.writerow([epoch+1, val_loss, val_acc, f1])

        if f1 > self.best_f1:
            self.best_f1 = f1; self.wait = 0
            self.model.save(self.filepath)
            print(f"✅ Improved macro F1. Model saved to {self.filepath}")
        else:
            self.wait += 1
            print(f"⚠️ No improvement in F1. Patience {self.wait}/{self.patience}")
            if self.wait >= self.patience:
                print("⛔ Early stopping (val_macro_f1).")
                self.model.stop_training = True

# =====================
# Training
# =====================
callbacks = [
    F1EarlyStopping(validation_data=(X_val, y_val),
                    filepath="best_cnn_bilstm_trainable.h5",
                    patience=5,
                    log_file="training_log.csv")
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# =====================
# Evaluation
# =====================
y_pred = model.predict(X_test, batch_size=32)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = y_test
y_confidence = np.max(y_pred, axis=1)

test_acc = accuracy_score(y_true_classes, y_pred_classes)
test_f1 = f1_score(y_true_classes, y_pred_classes, average="macro")
cm = confusion_matrix(y_true_classes, y_pred_classes)
report = classification_report(y_true_classes, y_pred_classes, target_names=[str(c) for c in le.classes_])

print("\n=== FINAL TEST RESULTS ===")
print(f"Accuracy: {test_acc*100:.2f}%")
print(f"Macro F1: {test_f1:.4f}")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)

# =====================
# Simpan hasil
# =====================
with open("results_trainable.txt", "w", encoding="utf-8") as f:
    f.write("=== FINAL TEST RESULTS ===\n")
    f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
    f.write(f"Test Macro F1: {test_f1:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")

# =====================
# Confusion Matrix Plot
# =====================
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[str(c) for c in le.classes_],
            yticklabels=[str(c) for c in le.classes_])
plt.xlabel("Prediksi"); plt.ylabel("Asli")
plt.title("Confusion Matrix - CNN–BiLSTM (Trainable FastText)")
plt.tight_layout()
plt.savefig("cnn_bilstm_trainable_cm.png")
plt.close()

# =====================
# Learning Curves
# =====================
log_df = pd.read_csv("training_log.csv")
plt.figure(figsize=(8,5))
plt.plot(log_df["epoch"], log_df["val_loss"], label="Val Loss")
plt.plot(log_df["epoch"], log_df["val_acc"], label="Val Accuracy")
plt.plot(log_df["epoch"], log_df["val_f1"], label="Val Macro F1")
plt.xlabel("Epoch"); plt.ylabel("Score")
plt.title("Learning Curves - CNN–BiLSTM (Trainable FastText)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("cnn_bilstm_trainable_learning_curves.png")
plt.show()
