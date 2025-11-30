
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
import os


# Cau hinh mo hinh va tham so
MODEL_NAME = "vinai/phobert-base-v2"
NUM_LABELS = 3
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "phobert_sentiment_model.pt"
ENCODER_SAVE_PATH = "label_encoder.json"
labels_name = ["negative", "neutral", "positive"]


# Dataset tùy chỉnh để chuẩn bị dữ liệu cho mô hình
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



def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Load model đúng cách, KHÔNG dùng .to() → tránh lỗi meta tensor
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        device_map={"": DEVICE},  # đặt toàn bộ model vào đúng device
        torch_dtype="auto"
    )

    # Nếu có model đã fine-tune → load
    if os.path.exists(MODEL_SAVE_PATH):
        state = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        model.load_state_dict(state)

    model.eval()
    return tokenizer, model

tokenizer, model = load_model()



# # Hàm dự đoán cảm xúc cho một đoạn văn bản
def predict_sentiment(text):
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()

    return labels_name[pred_id], probs[0].tolist()


# Hàm huấn luyện mô hình cảm xúc từ file CSV

def train_model(csv_file):
    # Đọc, làm sạch và lọc dữ liệu hợp lệ
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_file).dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].str.lower().str.strip()

    df = df[df["label"].isin(labels_name)]

    # Chuyển nhãn text -> số
    encoder = LabelEncoder()
    df["label_id"] = encoder.fit_transform(df["label"])

 
    import json
    with open(ENCODER_SAVE_PATH, "w", encoding="utf8") as f:
        json.dump(encoder.classes_.tolist(), f, ensure_ascii=False)

    # Chia dữ liệu train/val (stratify để giữ tỉ lệ nhãn)
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"],
        df["label_id"],
        test_size=0.2,
        stratify=df["label_id"],
        random_state=42
    )
    # Chuẩn bị batch dữ liệu cho train và validation
    train_loader = DataLoader(
        SentimentDataset(X_train.tolist(), y_train.tolist(), tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        SentimentDataset(X_val.tolist(), y_val.tolist(), tokenizer),
        batch_size=BATCH_SIZE
    )

      # Load PhoBERT để train 
    train_model_ft = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    ).to(DEVICE)

    # Optimizer và Scheduler cho mô hình Transformer
    optimizer = AdamW(train_model_ft.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    # Training
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        train_model_ft.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = train_model_ft(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        #VALIDATION
        train_model_ft.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = train_model_ft(**batch)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} — Train Loss: {avg_train_loss:.4f} — Val Loss: {avg_val_loss:.4f}")

        # Lưu model tốt nhất dựa trên val_loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(train_model_ft.state_dict(), MODEL_SAVE_PATH)
            print("Đã lưu model tốt nhất")

    print("Huấn luyện hoàn thành")




# Load model đã train
def load_trained_model():
    if os.path.exists(MODEL_SAVE_PATH):
        state = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        print("Đã load model đã huấn luyện")
    else:
        print("Chưa có model huấn luyện sẵn. Vui lòng train trước.")
