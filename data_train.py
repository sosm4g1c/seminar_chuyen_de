import random

# =========================
# Positive (tích cực)
# =========================
positive_phrases = [
    # Cảm xúc cá nhân
    "tôi rất vui", "mình cảm thấy hạnh phúc", "hôm nay thật tuyệt", "tôi thấy rất tốt",
    "cuộc sống thật đẹp", "tôi yêu công việc này", "mình đang rất hứng khởi",
    "tôi cực kỳ phấn khích", "thật là một ngày ý nghĩa", "mọi thứ đều ổn định và vui vẻ","toi vui", "toi thich", "toi hanh phuc"
    # Liên quan đến sản phẩm/dịch vụ
    "rất tốt", "hài lòng", "xuất sắc", "tuyệt vời", "ngon lành", "đáng đồng tiền",
    "dịch vụ tận tình", "rất thích", "vượt mong đợi", "sẽ ủng hộ lần sau",
    "đẹp và bền", "đáng giá", "cực kỳ ổn", "rất đáng mua", "mình khuyên nên thử"
]

# =========================
# Neutral (trung lập)
# =========================
neutral_phrases = [
    # Cảm xúc con người
    "tôi thấy bình thường", "ngày hôm nay cũng như mọi ngày", "không có gì đặc biệt","toi binh thuong", "toi khong cam thay gi nhieu","toi chi dang quan sat",
    "tôi không cảm thấy gì nhiều", "tôi chỉ đang quan sát", "không vui cũng không buồn",
    "thời tiết hôm nay bình thường", "tôi đang ở nhà", "máy tính của tôi màu đen",
    "tôi có một chiếc xe máy", "tôi đang nghe nhạc", "đây là một cái bàn",
    # Liên quan đến sản phẩm
    "bình thường", "không tệ", "vừa đủ dùng", "tạm được", "chấp nhận được",
    "ổn", "chưa ấn tượng", "cũng được", "tạm ổn", "được thôi",
    "trung bình", "khá ổn", "đúng như mong đợi", "tạm hài lòng"
]

# =========================
# Negative (tiêu cực)
# =========================
negative_phrases = [
    # Cảm xúc con người
    "tôi rất buồn", "mình cảm thấy thất vọng", "hôm nay thật tệ", "tôi chán nản","toi buon", "toi that vong", "toi met moi",
    "cuộc sống thật khó khăn", "tôi ghét công việc này", "mình đang rất mệt mỏi","that la mot ngay kinh khung", "toi ghet", "toi met moi", "toi phan no",
    "mọi thứ đều đi sai hướng", "tôi thấy bực mình", "cuộc sống thật mệt mỏi",
    "tôi giận dữ", "tôi không muốn nói chuyện nữa", "mình thấy vô vọng",
    # Liên quan đến sản phẩm/dịch vụ
    "thất vọng", "rất tệ", "hỏng", "không đúng mô tả", "không hài lòng",
    "tiền mất tật mang", "rất kém", "phải trả lại", "không đáng mua", "dịch vụ tệ",
    "đóng gói cẩu thả", "giao hàng chậm", "rất bực mình", "hoàn toàn thất vọng", "sản phẩm lỗi"
]

# =========================
# Chủ thể / nhân vật ngẫu nhiên
# =========================
subjects = [
    "Tôi", "Mình", "Bản thân tôi", "Tụi mình", "Cả nhóm", "Ngày hôm nay", "Thời tiết", 
    "Công việc", "Cuộc sống", "Sản phẩm", "Dịch vụ", "Hàng hóa", "Đơn hàng", "Món hàng",
    "Trải nghiệm này", "Buổi sáng nay", "Buổi tối hôm qua", "Cuộc gặp gỡ", "Chuyến đi", "Giao dịch"
]

# =========================
# Hàm tạo câu
# =========================
def generate_sentence(label):
    subj = random.choice(subjects)
    if label == "positive":
        phrase = random.choice(positive_phrases)
    elif label == "neutral":
        phrase = random.choice(neutral_phrases)
    else:
        phrase = random.choice(negative_phrases)
    
    ending = random.choice([".", "!", "!!", ""])
    return f"{subj} {phrase}{ending}"

# =========================
# Sinh dữ liệu và lưu
# =========================
data = {"text": [], "label": []}
for _ in range(1500):
    label = random.choice(["positive", "neutral", "negative"])
    sentence = generate_sentence(label)
    data["text"].append(sentence)
    data["label"].append(label)

import pandas as pd
df = pd.DataFrame(data)
df.to_csv("reviews_sentiment_2000_human_emotion.csv", index=False, encoding="utf-8-sig")

print("✅ Đã tạo file 'reviews_sentiment_2000_human_emotion.csv' gồm 2000 câu đa dạng với cảm xúc con người.")
