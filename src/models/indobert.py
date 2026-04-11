# ============================================
# 🚀 4. IndoBERT Training (AdamW Default) + Save Tokenizer lr = 1e-5 batch size = 16 
# ============================================
import os, re, string, csv
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =====================
# Setup Device
# =====================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================
# Hyperparameters
# =====================
LR          = 1e-5
BATCH_SIZE  = 16
NUM_EPOCHS  = 16
PATIENCE    = 3

# =====================
# Load Tokenizer & Model
# =====================
model_name = "indobenchmark/indobert-base-p2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6).to(device)

# =====================
# Tambah Special Tokens
# =====================
special_tokens_file = "/kaggle/input/specialtoken/special_tokens.txt"
if os.path.exists(special_tokens_file):
    with open(special_tokens_file, "r", encoding="utf-8") as f:
        special_tokens_list = [line.strip() for line in f.readlines() if line.strip()]
    
    if special_tokens_list:
        tokenizer.add_tokens(special_tokens_list)
        model.resize_token_embeddings(len(tokenizer))
        print(f"✅ {len(special_tokens_list)} special tokens ditambahkan ke tokenizer & model.")
    else:
        print("⚠️ File special_tokens.txt kosong.")
else:
    print("ℹ️ Tidak ada special_tokens.txt.")

# =====================
# Load Dataset
# =====================
train_df = pd.read_csv("/kaggle/input/undersample-u30k/train.csv")
val_df   = pd.read_csv("/kaggle/input/undersample-u30k/val.csv")
test_df  = pd.read_csv("/kaggle/input/undersample-u30k/test.csv")

for df in (train_df, val_df, test_df):
    df["normal_teks"] = df["normal_teks"].astype(str)
    df["Label_Validator"] = df["Label_Validator"].astype(int)

# Encode label
le = LabelEncoder()
train_df["Label_Validator"] = le.fit_transform(train_df["Label_Validator"])
val_df["Label_Validator"]   = le.transform(val_df["Label_Validator"])
test_df["Label_Validator"]  = le.transform(test_df["Label_Validator"])

# =====================
# Cleaning Function
# =====================
def clean_text(text):
    text = re.sub(r"(\w)([?!])", r"\1 \2", text)
    punct_to_remove = string.punctuation.replace("?", "").replace("!", "").replace("-", "")
    return text.translate(str.maketrans("", "", punct_to_remove)).lower()

for df in (train_df, val_df, test_df):
    df["normal_teks"] = df["normal_teks"].apply(clean_text)

# =====================
# Tokenization
# =====================
def tokenize_data(texts, labels, tokenizer, max_length=256):
    inputs = tokenizer(
        list(texts), padding=True, truncation=True,
        return_tensors="pt", max_length=max_length
    )
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long)
    }

train_data = tokenize_data(train_df["normal_teks"], train_df["Label_Validator"], tokenizer)
val_data   = tokenize_data(val_df["normal_teks"],   val_df["Label_Validator"], tokenizer)
test_data  = tokenize_data(test_df["normal_teks"],  test_df["Label_Validator"], tokenizer)

# =====================
# Dataset & Loader
# =====================
class CustomDataset(Dataset):
    def __init__(self, encodings, device):
        self.encodings = encodings
        self.device = device
    def __getitem__(self, idx):
        return {k: v[idx].clone().detach().to(self.device) for k,v in self.encodings.items()}
    def __len__(self):
        return len(self.encodings["labels"])

train_loader = DataLoader(CustomDataset(train_data, device), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(CustomDataset(val_data, device),   batch_size=BATCH_SIZE)
test_loader  = DataLoader(CustomDataset(test_data, device),  batch_size=BATCH_SIZE)

# =====================
# Training Setup
# =====================
loss_fn = nn.CrossEntropyLoss()

from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=LR)  # default weight_decay=0.01

save_dir = "runs_indobert(u30k)_versi4"
os.makedirs(save_dir, exist_ok=True)

# simpan tokenizer untuk inference
tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))
print("📦 Tokenizer saved!")

best_f1, best_epoch, epochs_no_improve = 0, -1, 0

log_file = os.path.join(save_dir, "training_log.csv")
with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_f1", "val_loss", "val_acc", "val_f1"])

# =====================
# Evaluate function
# =====================
def evaluate_model(model, data_loader, device, loss_fn):
    model.eval()
    preds, targets, probs = [], [], []
    total_loss, total_samples = 0.0, 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            logits = outputs.logits
            loss = loss_fn(logits, batch["labels"])

            total_loss += loss.item() * batch["labels"].size(0)
            total_samples += batch["labels"].size(0)

            batch_probs = torch.softmax(logits, dim=1)
            preds.extend(torch.argmax(batch_probs, dim=1).cpu().numpy())
            targets.extend(batch["labels"].cpu().numpy())
            probs.extend(batch_probs.cpu().numpy())

    avg_loss = total_loss / total_samples
    return np.array(targets), np.array(preds), np.array(probs), avg_loss

# =====================
# Training Loop
# =====================
for epoch in range(NUM_EPOCHS):
    print(f"\n=== Epoch {epoch+1}/{NUM_EPOCHS} ===")
    model.train()
    total_loss, correct, total = 0, 0, 0
    train_preds, train_targets = [], []

    for batch in tqdm(train_loader, desc="Training", leave=False):
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"], 
            attention_mask=batch["attention_mask"]
        )
        logits = outputs.logits
        loss = loss_fn(logits, batch["labels"])

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch["labels"].size(0)

        preds = torch.argmax(logits, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_targets.extend(batch["labels"].cpu().numpy())

        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)

    train_avg_loss = total_loss / total
    train_acc = correct / total
    train_f1 = f1_score(train_targets, train_preds, average="macro")

    print(f"[Train] Loss={train_avg_loss:.4f}, Acc={train_acc*100:.2f}%, F1={train_f1:.4f}")

    val_labels, val_preds, _, val_loss = evaluate_model(model, val_loader, device, loss_fn)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average="macro")

    print(f"[Val] Loss={val_loss:.4f}, Acc={val_acc*100:.2f}%, F1={val_f1:.4f}")

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, train_avg_loss, train_f1, val_loss, val_acc, val_f1])

    if val_f1 > best_f1:
        best_f1, best_epoch, epochs_no_improve = val_f1, epoch+1, 0
        torch.save(model.state_dict(), f"{save_dir}/best_model.pt")
        print("✅ Improved! Best model saved.")
    else:
        epochs_no_improve += 1
        print(f"⚠ No improvement ({epochs_no_improve}/{PATIENCE})")
        if epochs_no_improve >= PATIENCE:
            print("⛔ Early Stopping triggered")
            break

# =====================
# Final Test Evaluation
# =====================
model.load_state_dict(torch.load(f"{save_dir}/best_model.pt"))

test_labels, test_preds, test_probs, test_loss = evaluate_model(model, test_loader, device, loss_fn)

test_acc = accuracy_score(test_labels, test_preds)
test_f1  = f1_score(test_labels, test_preds, average="macro")
cm = confusion_matrix(test_labels, test_preds)
report = classification_report(test_labels, test_preds, digits=4)

print("\n=== FINAL TEST RESULTS ===")
print(f"Accuracy: {test_acc*100:.2f}%")
print(f"Macro F1: {test_f1:.4f}")
print("Confusion Matrix:\n", cm)
print(report)

wrong_mask = (test_preds != test_labels)
wrong_df = test_df.loc[wrong_mask, ["Indeks", "normal_teks", "Label_Validator"]].copy()
wrong_df["Prediksi_model"] = test_preds[wrong_mask]
wrong_df["confidence"] = test_probs[wrong_mask].max(axis=1)

wrong_df.to_csv(os.path.join(save_dir, "wrong_predictions.csv"), index=False)

# =====================
# Plot Learning Curves
# =====================
log_df = pd.read_csv(log_file)

plt.figure(figsize=(8,5))
plt.plot(log_df["epoch"], log_df["train_loss"], label="Train Loss")
plt.plot(log_df["epoch"], log_df["val_loss"], label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Train vs Val Loss")
plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "loss_curve.png"))
plt.show()

plt.figure(figsize=(8,5))
plt.plot(log_df["epoch"], log_df["train_f1"], label="Train F1")
plt.plot(log_df["epoch"], log_df["val_f1"], label="Val F1")
plt.xlabel("Epoch"); plt.ylabel("F1 Score")
plt.title("Train vs Val Macro-F1")
plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "f1_curve.png"))
plt.show()
