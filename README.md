# knn-face_recognition
B1: Tạo train_dir và test_dir theo cấu trúc sau:

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
B2: Thêm khuôn mặt muốn nhận diện vào tập train_dir và test_dir theo cấu trúc như B1
- Train_dir: Thư mục bao gồm thư mực con cho mỗi người được đưa vào để nhận dạng với tên của họ.
Mỗi thư mục con chứa các tệp hình ảnh chứa khuôn mặt của người đó, mục đích để phục vụ cho việc huấn luyện model
- Test_dir : Cấu trúc giống bộ train_dir để lưu trữ mục chứa tập dữ liệu kiểm tra để đánh giá độ chính xác của mô
hình nhận dạng khuôn mặt, mục đích để lấy các nhãn trong thư mục này tính toán độ chính xác của mô hình bằng cách
so sánh tên nhãn được dự đoán bởi mô hình và tên nhãn thực tế trong bộ dữ liệu kiểm tra (test_dir)
B3: Chạy file hello1.py
B4: Click PC camera và chương trình bắt đầu nhận diện theo thời gian thực
 - Nếu khuôn mặt có trong tập train và test thì chương trình sẽ nhận diện được tên và độ chính xác
 - Nếu khuôn mặt không có trong dập train và test thì chương trình sẽ trả về 'unknown'.
