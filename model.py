# model.py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
torch.autograd.set_detect_anomaly(True)
torch.use_deterministic_algorithms(False)
import pandas as pd
import os

# =========================
# Cấu hình
# =========================
MODEL_NAME = "vinai/phobert-base-v2"
NUM_LABELS = 3
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "phobert_sentiment_model.pt"

# =========================
# Dataset
# =========================
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# =========================
# Load tokenizer & model
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model.to(DEVICE)

labels_name = ["negative", "neutral", "positive"]

# =========================
# Dự đoán
# =========================
def predict_sentiment(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs).item()
    return labels_name[pred_id], probs[0].tolist()

# =========================
# Huấn luyện model
# =========================
def train_model(csv_file):
    model.train()
    model.zero_grad(set_to_none=True)
    # Load dữ liệu
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(labels_name)]
    
    # Mã hóa nhãn
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    df["label_id"] = encoder.fit_transform(df["label"])
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"], df["label_id"], test_size=0.2, stratify=df["label_id"], random_state=42
    )

    train_dataset = SentimentDataset(X_train.tolist(), y_train.tolist(), tokenizer, MAX_LEN)
    val_dataset = SentimentDataset(X_val.tolist(), y_val.tolist(), tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Huấn luyện
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Train loss: {avg_loss:.4f}")
    
    # Lưu model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Đã lưu model vào {MODEL_SAVE_PATH}")

# =========================
# Load model đã train
# =========================
def load_trained_model():
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        model.eval()
        print("Đã load model đã huấn luyện")
    else:
        print("Chưa có model huấn luyện sẵn. Vui lòng train trước.")
