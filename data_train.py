import random
import pandas as pd

# =========================
# 500 câu neutral (không dấu)
# =========================
neutral_phrases = [
"hom nay toi di hoc","toi vua an trua","toi dang o nha","toi sap di lam","toi vua ve toi",
"toi dang cho xe bus","hom nay troi nang","ngoai troi dang mua","toi dang doi ban","toi dang nghi ngoi",
"toi vua doc sach","toi dang nghe nhac","toi vua don phong","toi moi di cho ve","toi dang tap the duc",
"toi vua uong nuoc","toi dang xem tivi","toi dang xem phim","toi sap di ngu","toi vua ngu day",
"toi sap ra ngoai","toi co hen voi ban","toi dang gap giao thong","duong dang ket","hom nay toi co tiet som",
"toi dang nghi phep","toi dang doi den xanh","may tinh dang cap nhat","wifi nha toi binh thuong","toi dang doi email",
"toi chua an sang","toi moi an sang xong","toi nho dem do","toi sap nop bai","toi dang lam bai tap",
"toi dang hop online","toi chuan bi hop","hop chua bat dau","toi vua tan hoc","noi lam viec dang dong",
"ngoai duong dang dong nguoi","toi dang xep hang","hom nay thu hai","hom nay thu ba","hom nay thu tu",
"hom nay thu nam","hom nay thu sau","hom nay thu bay","hom nay chu nhat","toi dang xuat ban excel",
"toi dang ghi chu","toi dang doc tai lieu","toi vua kiem tra mail","dien thoai sap het pin","toi dang sac pin",
"toi vua goi dien xong","toi dang cho tin nhan","toi dang doi phan hoi","toi vua tra loi tin nhan","toi dang doi cap nhat",
"toi dang kiem tra he thong","toi dang lam bien ban","toi dang hoan thanh form","toi dang doi xe grab","toi sap toi noi",
"toi dang o ben xe","xe bus sap den","toi dang o cua hang","toi dang chon hang","toi vua thanh toan",
"toi dang doi hoa don","toi dang giao hang","hang dang van chuyen","toi dang doi shipper","hom nay troi mat",
"thoi tiet kha on","gio dang thoi nhe","nhiet do kha cao","nhiet do kha thap","toi dang kiem tra thoi tiet",
"toi dang cho giang vien","toi dang doi lop truoc ra","toi da o truoc cua","lop sap bat dau","lop vua tan",
"toi vua xong viec","toi dang lam bao cao","file dang upload","file dang download","may anh dang sac",
"toi dang chon do an","toi chua quyet dinh","toi se di mua do","toi dang ve nha","toi dang doi tin tuc",
"toi moi nhan duoc thong bao","toi dang xem lich","toi dang sap xep lich","toi dang xem lai cong viec","toi dang de nghi duyet",
"toi dang cho duyet","he thong dang khoi dong","toi dang lam thu tuc","toi dang doi thu tuc","toi dang cho lay so",
"toi dang doi lay hang","toi dang cho lay thuoc","toi dang doi lay giay to","toi dang tim dia chi","toi dang xem ban do",
"toi dang nhap du lieu","toi dang nhap tai khoan","dang cho xac thuc","toi dang kiem tra man hinh","toi dang xoay man hinh",
"toi dang doi xe chay","xe dang dung den do","toi dang di bo","toi dang tap di bo","toi dang di dao",
"toi dang doa rac","toi dang giat do","toi dang phoi do","toi dang don bep","toi dang nau an",
"toi dang roi nha","toi dang o san thuong","toi dang xuong tang 1","toi dang cho thang may","toi dang o thang may",
"thang may dang di len","thang may dang di xuong","toi dang cat toc","toi dang rua mat","toi dang danh rang",
"toi dang thay quan ao","toi dang gap quan ao","toi dang thay pin remote","toi dang do pin","toi dang mo ui do",
"toi dang gap do","toi dang quet nha","toi dang lau nha","toi dang giao bai tap","toi vua nop file",
"toi dang xem diem","toi dang cho ket qua","toi dang luu file","toi dang sap xep tai lieu","toi dang sua file",
"toi dang doi update","toi dang backup file","toi dang kiem tra drive","toi dang xem video","toi dang xem livestream",
"toi dang ket noi bluetooth","toi dang tim wifi","toi dang kiem tra modem","toi dang mo ung dung","toi dang tat ung dung",
"toi dang cai dat ung dung","toi dang xoa ung dung","toi dang restart may","toi dang tat may","toi dang bat may",
"toi dang ve que","toi dang di duong","toi dang doi truc","toi dang doi nhan su","toi dang kiem tra hop dong",
"toi dang doc hop dong","toi dang doi ky","toi dang xem thong tin","dang cho xac minh","toi dang kiem tra so lieu",
"toi dang tim tai lieu","toi dang chuyen file","toi dang nop ho so","toi dang doi goi hang","toi dang xuat bang",
"toi dang nhap bang","toi dang lam bieu mau","toi dang chuan bi tai lieu","toi dang doi trinh ky","toi dang check lich",
"toi dang doi sap xep","toi dang doi phien","toi dang doi den","toi dang doi tau","toi dang o ga tau",
"tau sap den","toi dang tim ghe","toi dang tim duong","toi dang doi may bay","toi dang doi check in",
"toi dang doi len may bay","toi dang doi lay ve","toi dang doi hanh ly","toi dang xem lai vi","toi dang dem tien",
"toi dang doi tien le","toi dang doi the","toi dang o ngan hang","toi dang rut tien","toi dang chuyen tien",
"toi dang kiem tra so du","toi dang o quan ca phe","toi dang go phim","toi dang doi ban den","toi dang cho order",
"toi dang doi pha che","toi dang ngoai ban cong","toi dang trong phong khach","toi dang tam","toi dang rua chen",
"toi dang phoi quan ao","toi dang cho do kho","toi dang chup anh","toi dang quay video","toi dang ghi am",
"toi dang nhan hang","toi dang ki nhan","toi dang mo thung","toi dang kiem tra hang","toi dang don do",
"toi dang sap xep phong","toi dang sua bong den","toi dang sua o dien","toi dang bat den","toi dang tat den",
"toi dang bat quat","toi dang tat quat","toi dang mo cua","toi dang dong cua","toi dang xep lich hop",
"toi dang xep lich hoc","toi dang chen giay to","toi dang thay muc in","toi dang ket noi may in","toi dang cho in",
"may in dang chay","may in dang ket noi","toi dang doi tai lieu ra","toi dang ngoi doi","toi dang xem may chieu",
"toi dang doi thay pin chuot","toi dang thu am","toi dang chuan bi bai","toi dang chuan bi do","toi dang sap xep ban",
"toi dang xem danh sach","toi dang doi xac nhan"
]

# =========================
# Chủ thể (để câu tự nhiên hơn)
# =========================
subjects = [
    "toi", "minh", "ban", "chung toi", "ca nhom", "hom nay", "thoi tiet",
    "cong viec", "buoi sang nay", "buoi chieu nay", "toi luc nay", "hien tai"
]

# =========================
# Hàm tạo câu neutral
# =========================
def generate_sentence():
    subj = random.choice(subjects)
    phrase = random.choice(neutral_phrases)
    ending = random.choice([".", "", ".", ""])
    return f"{subj} {phrase}{ending}".strip()

# =========================
# Sinh dữ liệu và lưu
# =========================
data = {"text": [], "label": []}

for _ in range(500):
    sentence = generate_sentence()
    data["text"].append(sentence)
    data["label"].append("neutral")

df = pd.DataFrame(data)
df.to_csv("neutral_500.csv", index=False, encoding="utf-8-sig")

print("✅ Da tao file 'neutral_500.csv' voi 500 cau neutral khong dau.")
