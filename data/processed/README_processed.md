# Dữ liệu nhiệt độ đã xử lý

Thư mục này lưu các file kết quả được tạo bởi `preprocess_temperature_pipeline()`.

Các file dự kiến được tạo:

- `temperature_clean.csv`: chuỗi thời gian theo ngày đã được làm sạch, dùng tên cột chuẩn `date` và `temperature`.
- `temperature_train_scaled.csv`: tập train được chia theo thứ tự thời gian, có cột `temperature_scaled`.
- `temperature_val_scaled.csv`: tập validation được chia theo thứ tự thời gian, có cột `temperature_scaled`.
- `temperature_test_scaled.csv`: tập test được chia theo thứ tự thời gian, có cột `temperature_scaled`.
- `temperature_scaled.csv`: toàn bộ dữ liệu đã chuẩn hóa, có cột `split`.
- `split_info.json`: thông tin kích thước các tập dữ liệu và khoảng ngày tương ứng.

Quy tắc chuẩn hóa:

- Chỉ tính tham số chuẩn hóa trên tập train.
- Áp dụng giá trị trung bình và độ lệch chuẩn của tập train cho tập validation và test.
- Lưu tham số chuẩn hóa vào `models/scaler_params.json`.

Quy tắc chuỗi thời gian:

- Không shuffle dữ liệu.
- Giữ nguyên thứ tự thời gian cho train, validation và test.
