# 3121560031 - BẠCH NGUYỄN HỮU HIỆU - BÁO CÁO ĐỒ ÁN CUỐI KÌ - CHUYÊN ĐỀ SEMINAR 


## Tên đồ án: XÂY DỰNG TRỢ LÝ PHÂN LOẠI CẢM XÚC TIẾNG VIỆT SỬ DỤNG TRANSFORMER


## Thành viên nhóm:
| Thành viên | Mã số sinh viên |
| :------- | :------: |
| Bạch Nguyễn Hữu Hiệu | 3121560031  |

## 1. Cấu trúc thư mục đồ án:
* app.py : Chứa các đoạn code hiển thị giao diện ứng dụng
* model.py : Chứa các đoạn code để sử dụng model PhoBERT và thực hiện training cho mô hình
* data_train.py: Chứa các đoạn tạo file csv dữ liệu tự động cho việc thực hiện huấn luyện cho mô hình
* sentiments.db: File cơ sở dữ liệu SQLite chứa các câu đã được phân tích từ ứng dụng
* Các file csv còn lại: Là các file dữ liệu để huấn luyến mô hình

## 2. Hướng dẫn cài đặt và sử dụng:
### 2.1. Yêu cầu hệ thống:
**Yêu cầu phần cứng:** 
* RAM tối thiểu: 4GB 
* Bộ nhớ trống tối thiểu: 2GB 

**Yêu cầu hệ thống:**
Để cài đặt được ứng dụng bạn cần phải cài sẵn các phần mềm sau: 
* Python: Phiên bản từ 3.9 -> 3.12
* Pip: Trình quản lý đi kèm với Pythong dùng để cài đặt các thư viện cần thiết
* Git: Công cụ quản lý phiên bản dùng để sao chép mã nguồn của đồ án
* SQLite3: Hệ quản trị cơ sở dữ liệu nhúng, được Python hỗ trợ sẵn qua module sqlite3. Cài đặt extension SQLite trên visual studio code để theo dõi dữ liệu.

**Các thư viện được sử dụng trong ứng dụng:** 
* **Tourch**: Thư viện nền tảng cho mô hình học sâu:
```php

pip install torch

```
* **Transformers**: Dùng để tải và chạy mô hình PhoBERT.
```php

pip install transformers 

```
* **Underthesea**: Thư viện xử lý tiếng Việt (tokenize, POS tagging):
```php

pip install underthesea  

```
* **Streamlit**: Tạo giao diện web cho ứng dụng:
```php

pip install streamlit  

```
* **Pandas**: Xử lý dữ liệu CSV và quản lý dataframe:
```php

pip install pandas 

```
* **scikit-learn**: Hỗ trợ encode nhãn, chia tập dữ liệu train/test:
```php

pip install scikit-learn 

```
---
### 2.2. Cài đặt:
#### B1. Sao chép mã nguồn:
##### Mở terminal hoặc command prompt và chạy lệnh sau để sao chép repository từ GitHub:
```php

git clone https://github.com/sosm4g1c/ seminar_chuyen_de.git

```
#### B2. Di chuyển vào thư mục dự án:
```php

Cd seminar_chuyen_de 

```
#### B3. Cài đặt các thư viện Python cần thiết (Có thể bỏ qua nếu đã cài đặt sẵn các thư viện theo yêu cầu phía trên)
```php

pip install torch transformers underthesea streamlit pandas scikit-learn 

```
#### B4. Tạo cơ sở dữ liệu SQLite:
##### File SQLite sẽ được tạo tự động khi chạy ứng dụng lần đầu. Không cần thao tác gì thêm, nhưng đảm bảo bạn có quyền ghi file trong thư mục dự án.
#### B5. Chạy ứng dụng:
##### Khởi động ứng dụng Streamlit bằng lệnh: 
```php

streamlit run app.py  

```
##### Mở trình duyệt và truy cập địa chỉ: 
```php

http://localhost:8501  

```
---
### 2.3. Tạo file dữ liệu để training:
#### Để tạo file csv training tự động cho mô hình, bạn cần chạy câu lệnh:
```php

python data_train.py 

```
#### Sau khi chạy câu lệnh 1 file csv chứa cơ sở dữ liệu định dạng theo 2 cột "text","label" xuất hiện trên cấu trúc thư mục.
---
### 2.4. Hướng dẫn sử dụng:
#### Sau khi chạy ứng dụng thành công, truy cập theo đường dẫn đã được cấu hình, giao diện website ứng dụng hiển thị:




