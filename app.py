import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from underthesea import word_tokenize
from torch.optim import AdamW
import pandas as pd
import sqlite3
import os
from datetime import datetime

# =========================
# CẤU HÌNH
# =========================
MODEL_NAME = "vinai/phobert-base-v2"
MODEL_SAVE_PATH = "phobert_sentiment_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
LABELS = ["negative", "neutral", "positive"]


# Từ điển chuyển từ không dấu hoặc viết tắt sang chuẩn
NORMALIZE_DICT = {
    "rat": "rất",
    "tot": "tốt",
    "k": "không",
    "ko": "không",
    "tui": "tôi",
    "dc": "được",
    "mn": "mọi người",
    "buon": "buồn",
    "vui": "vui",
    "ghen": "ghen",
    "thich": "thích"
}

# =========================
# KẾT NỐI SQLITE
# =========================
def init_db():
    conn = sqlite3.connect("sentiments.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            sentiment TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_record(text, sentiment):
    conn = sqlite3.connect("sentiments.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)",
        (text, sentiment, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

def fetch_all_records():
    conn = sqlite3.connect("sentiments.db")
    df = pd.read_sql_query("SELECT * FROM sentiments ORDER BY id DESC", conn)
    conn.close()
    return df

# =========================
# DATASET TRAINING
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
        encoding = tokenizer(
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
# LOAD TOKENIZER & MODEL
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS))
if os.path.exists(MODEL_SAVE_PATH):
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =========================
# TIỀN XỬ LÝ CÂU
# =========================
def preprocess_text(text):
    # 1️⃣ Chuyển chữ thường
    text = text.lower().strip()

    # 2️⃣ Thay các từ trong từ điển NORMALIZE_DICT
    for k, v in NORMALIZE_DICT.items():
        text = text.replace(k, v)

    # 3️⃣ Tách từ bằng underthesea
    text = " ".join(word_tokenize(text))

    # 4️⃣ Giới hạn độ dài <51 ký tự để giảm thời gian xử lý
    if len(text) > 50:
        text = text[:50]
    
    return text


# =========================
# HÀM DỰ ĐOÁN
# =========================
def predict_sentiment(text):
    text = preprocess_text(text)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs).item()
    return LABELS[pred_id], probs[0][pred_id].item()

# =========================
# HÀM TRAIN MODEL
# =========================
def train_model(csv_file):
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(LABELS)]

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
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    progress = st.progress(0)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress.progress((i + 1) / len(train_loader))
        st.write(f"✅ Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    st.success("🎉 Đã train xong và lưu model!")

# =========================
# GIAO DIỆN STREAMLIT
# =========================
init_db()
st.title("💬 Ứng dụng phân tích cảm xúc tiếng Việt (PhoBERT)")

tab1, tab2, tab3 = st.tabs(["📊 Dự đoán", "🧠 Huấn luyện model", "🗂️ Lịch sử lưu trữ"])

# ----------------------
# TAB 1: Dự đoán cảm xúc
# ----------------------
with tab1:
    user_input = st.text_area("Nhập câu cần phân tích cảm xúc:", "")
    if st.button("Phân tích"):
        if not user_input.strip() or len(user_input.strip()) < 5:
            st.error("⚠️ Câu không hợp lệ, thử lại.")
        else:
            label, prob = predict_sentiment(user_input)
            if label == "positive":
                st.success(f"Kết quả: **{label.upper()}** (Độ tin cậy: {prob:.2f})")
            elif label == "negative":
                st.error(f"Kết quả: **{label.upper()}** (Độ tin cậy: {prob:.2f})")
            else:
                st.info(f"Kết quả: **{label.upper()}** (Độ tin cậy: {prob:.2f})")
            insert_record(user_input, label)
            st.info("✅ Đã lưu kết quả vào cơ sở dữ liệu!")

# ----------------------
# TAB 2: Train model
# ----------------------
with tab2:
    st.write("Tải lên file CSV để huấn luyện model:")
    csv_file = st.file_uploader("Chọn file .csv", type=["csv"])
    if csv_file is not None:
        with open("train_data.csv", "wb") as f:
            f.write(csv_file.getbuffer())
        if st.button("Bắt đầu huấn luyện"):
            train_model("train_data.csv")

# ----------------------
# TAB 3: Hiển thị lịch sử
# ----------------------
with tab3:
    st.subheader("🧾 Lịch sử dự đoán đã lưu:")
    data = fetch_all_records()
    if not data.empty:
        st.dataframe(data)
    else:
        st.info("Chưa có dữ liệu được lưu.")
