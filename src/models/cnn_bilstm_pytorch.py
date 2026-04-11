# -*- coding: utf-8 -*-
# Multi-kernel CNN–BiLSTM in PyTorch (6 classes)
# ------------------------------------------------------------
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =====================
# Load Dataset
# =====================
train_df = pd.read_csv("Data/train.csv")
val_df   = pd.read_csv("Data/val.csv")
test_df  = pd.read_csv("Data/test.csv")

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

# Encode label
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
# Tokenization (pakai Keras Tokenizer biar gampang)
# =====================
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

max_len = 256
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(train_df["normal_teks"])
max_words = len(tokenizer.word_index) + 1

X_train = pad_sequences(tokenizer.texts_to_sequences(train_df["normal_teks"]), maxlen=max_len, padding="post")
X_val   = pad_sequences(tokenizer.texts_to_sequences(val_df["normal_teks"]),   maxlen=max_len, padding="post")
X_test  = pad_sequences(tokenizer.texts_to_sequences(test_df["normal_teks"]),  maxlen=max_len, padding="post")

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
    "Data/cc.id.300.vec/cc.id.300.vec",
    tokenizer.word_index,
    embed_dim=300,
    vocab_size=max_words
)

# =====================
# Dataset & DataLoader
# =====================
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(TextDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader   = DataLoader(TextDataset(X_val, y_val), batch_size=32, shuffle=False)
test_loader  = DataLoader(TextDataset(X_test, y_test), batch_size=32, shuffle=False)

# =====================
# Model Definition
# =====================
class MultiKernelCNNBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, embedding_matrix, num_classes,
                 conv_filters=128, lstm_units=128, dense_units=64, drop_rate=0.5):
        super(MultiKernelCNNBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.data.copy_(torch.tensor(embedding_matrix))
        self.embedding.weight.requires_grad = True

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=conv_filters, kernel_size=k, padding="same")
            for k in [3, 4, 5]
        ])
        self.lstm = nn.LSTM(input_size=conv_filters*3, hidden_size=lstm_units,
                            bidirectional=True, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(lstm_units*2, dense_units)
        self.dropout = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(dense_units, num_classes)

    def forward(self, x):
        x = self.embedding(x)             # (B, L, E)
        x = x.permute(0, 2, 1)            # (B, E, L)
        convs = [torch.max(torch.relu(conv(x)), dim=2)[0] for conv in self.convs]
        x = torch.cat(convs, dim=1)       # (B, conv_filters*3)
        x = x.unsqueeze(1)                # (B, 1, conv_filters*3)
        _, (h, _) = self.lstm(x)          # ambil hidden state terakhir
        h = torch.cat((h[-2,:,:], h[-1,:,:]), dim=1)   # (B, 2*lstm_units)
        x = torch.relu(self.fc1(h))
        x = self.dropout(x)
        out = self.fc2(x)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiKernelCNNBiLSTM(max_words, 300, embedding_matrix, num_classes).to(device)
print(model)

# =====================
# Training
# =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

best_f1 = -np.Inf
patience, wait = 5, 0
log_file = "training_log.csv"
with open(log_file, "w", newline="") as f:
    writer = csv.writer(f); writer.writerow(["epoch", "val_loss", "val_acc", "val_f1"])

for epoch in range(1, 51):
    model.train()
    train_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_preds, val_true = [], []
    val_loss = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(y.cpu().numpy())

    val_acc = accuracy_score(val_true, val_preds)
    val_f1 = f1_score(val_true, val_preds, average="macro")

    print(f"Epoch {epoch}: val_loss={val_loss/len(val_loader):.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f); writer.writerow([epoch, val_loss/len(val_loader), val_acc, val_f1])

    # Early stopping
    if val_f1 > best_f1:
        best_f1, wait = val_f1, 0
        torch.save(model.state_dict(), "best_multikernel_cnn_bilstm.pt")
        print("✅ Improved F1, model saved.")
    else:
        wait += 1
        print(f"⚠️ No improvement. Patience {wait}/{patience}")
        if wait >= patience:
            print("⛔ Early stopping triggered.")
            break

# =====================
# Evaluation
# =====================
model.load_state_dict(torch.load("best_multikernel_cnn_bilstm.pt"))
model.eval()
y_true, y_pred, y_conf = [], [], []
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        outputs = torch.softmax(model(X), dim=1)
        preds = torch.argmax(outputs, dim=1)
        confs = torch.max(outputs, dim=1)[0]
        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_conf.extend(confs.cpu().numpy())

test_acc = accuracy_score(y_true, y_pred)
test_f1 = f1_score(y_true, y_pred, average="macro")
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=[str(c) for c in le.classes_])

print("\n=== FINAL TEST RESULTS ===")
print(f"Accuracy: {test_acc*100:.2f}%")
print(f"Macro F1: {test_f1:.4f}")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)

with open("results.txt", "w", encoding="utf-8") as f:
    f.write("=== FINAL TEST RESULTS ===\n")
    f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
    f.write(f"Test Macro F1: {test_f1:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")

# =====================
# Save misclassified
# =====================
def save_misclassified(df, X, y_true, model, filename, name="Dataset"):
    dataset = TextDataset(X, y_true)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds, all_confs = [], []
    model.eval()
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            outputs = torch.softmax(model(Xb), dim=1)
            preds = torch.argmax(outputs, dim=1)
            confs = torch.max(outputs, dim=1)[0]
            all_preds.extend(preds.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())

    results = pd.DataFrame({
        "Indeks": df["Indeks"].values,
        "Komentar": df["normal_teks"].astype(str).values,
        "Label_Asli": y_true.astype(int),
        "Label_Prediksi": np.array(all_preds, dtype=int),
        "Confidence": np.array(all_confs)
    })

    mis = results[results["Label_Asli"] != results["Label_Prediksi"]]
    mis.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"{name}: {len(mis)}/{len(df)} salah prediksi ({len(mis)/len(df)*100:.2f}%) → {filename}")

# Train/Val/Test
save_misclassified(train_df, X_train, y_train, model, "multikernel_train_misclassified.csv", name="Train")
save_misclassified(val_df, X_val, y_val, model, "multikernel_val_misclassified.csv", name="Validation")
save_misclassified(test_df, X_test, y_test, model, "multikernel_test_misclassified.csv", name="Test")

# =====================
# Confusion Matrix Plot
# =====================
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[str(c) for c in le.classes_],
            yticklabels=[str(c) for c in le.classes_])
plt.xlabel("Prediksi"); plt.ylabel("Asli")
plt.title("Confusion Matrix - Multi-kernel CNN–BiLSTM (PyTorch)")
plt.tight_layout(); plt.savefig("multikernel_cnn_bilstm_cm.png"); plt.close()
