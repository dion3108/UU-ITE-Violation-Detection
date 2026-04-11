# ============================================================
# 🚀 BASELINE: IndoBERT Frozen (Feature Extraction) 
# Fokus: Deteksi UU ITE (6 Label) - Early Stopping & F1-Score
# Link Paper : https://ejournal.instiki.ac.id/index.php/sintechjournal/article/view/1846/553 
# ============================================================

import os, re, string, csv, torch
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# 1. Setup Device & Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "indobenchmark/indobert-base-p2"
LR = 1e-3  
BATCH_SIZE = 16
NUM_EPOCHS = 15 
PATIENCE = 3  # Early Stopping: berhenti jika 3 epoch tidak ada kenaikan F1
save_dir = "baseline_indobert_uuite"
os.makedirs(save_dir, exist_ok=True)

# 2. Load Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_base = AutoModel.from_pretrained(model_name)

# --- STRATEGI PAPER: FREEZING ---
for param in bert_base.parameters():
    param.requires_grad = False

class IndoBERT_Baseline(nn.Module):
    def __init__(self, bert_model, num_labels=6):
        super(IndoBERT_Baseline, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(768, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

model = IndoBERT_Baseline(bert_base, num_labels=6).to(device)

# 3. Preprocessing & Data Loading
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"(\w)([?!])", r"\1 \2", text)
    punct_to_remove = string.punctuation.replace("?", "").replace("!", "").replace("-", "")
    return text.translate(str.maketrans("", "", punct_to_remove))

train_df = pd.read_csv("/kaggle/input/undersample-u30k/train.csv")
val_df   = pd.read_csv("/kaggle/input/undersample-u30k/val.csv")
test_df  = pd.read_csv("/kaggle/input/undersample-u30k/test.csv")

for df in [train_df, val_df, test_df]:
    df["normal_teks"] = df["normal_teks"].astype(str).apply(clean_text)
    df["Label_Validator"] = df["Label_Validator"].astype(int)

le = LabelEncoder()
train_df["label_encoded"] = le.fit_transform(train_df["Label_Validator"])
val_df["label_encoded"]   = le.transform(val_df["Label_Validator"])
test_df["label_encoded"]  = le.transform(test_df["Label_Validator"])

class UUITEDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.encodings = tokenizer(list(texts), truncation=True, padding="max_length", 
                                   max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels.values, dtype=torch.long)
    def __getitem__(self, idx):
        item = {k: v[idx].to(device) for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx].to(device)
        return item
    def __len__(self): return len(self.labels)

train_loader = DataLoader(UUITEDataset(train_df["normal_teks"], train_df["label_encoded"], tokenizer), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(UUITEDataset(val_df["normal_teks"], val_df["label_encoded"], tokenizer), batch_size=BATCH_SIZE)
test_loader = DataLoader(UUITEDataset(test_df["normal_teks"], test_df["label_encoded"], tokenizer), batch_size=BATCH_SIZE)

# 4. Training Loop with Early Stopping
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

best_f1 = 0
epochs_no_improve = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = loss_fn(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch["input_ids"], batch["attention_mask"])
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().numpy())
    
    val_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")
    
    # Logic Early Stopping & Save Best Model
    if val_f1 > best_f1:
        best_f1 = val_f1
        epochs_no_improve = 0
        torch.save(model.state_dict(), f"{save_dir}/best_baseline_model.pt")
        print("⭐ F1 Meningkat! Model disimpan.")
    else:
        epochs_no_improve += 1
        print(f"ℹ️ Tidak ada peningkatan ({epochs_no_improve}/{PATIENCE})")
        
    if epochs_no_improve >= PATIENCE:
        print("🛑 Early Stopping dipicu. Berhenti melatih.")
        break

# 5. Final Report
model.load_state_dict(torch.load(f"{save_dir}/best_baseline_model.pt"))
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        logits = model(batch["input_ids"], batch["attention_mask"])
        test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        test_labels.extend(batch["labels"].cpu().numpy())

target_names = [str(c) for c in le.classes_]
report = classification_report(test_labels, test_preds, target_names=target_names, digits=4)
cm = confusion_matrix(test_labels, test_preds)

with open(os.path.join(save_dir, "hasil_baseline_summary.txt"), "w") as f:
    f.write(f"MODEL BASELINE (FROZEN)\n\nReport:\n{report}\n\nConfusion Matrix:\n{cm}")

print("✅ Selesai! Model baseline terbaik siap digunakan.")