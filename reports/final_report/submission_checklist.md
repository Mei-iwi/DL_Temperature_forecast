# Checklist nộp bài

- [x] README đã hoàn thiện.
- [x] `requirements.txt` đã hoàn thiện.
- [x] `main.py` hỗ trợ các chế độ CLI: `self_test`, `preprocess`, `train`, `evaluate`, `predict_next_day`, `run_all`.
- [x] `app_temperature_cli.py` hỗ trợ các chế độ CLI tương tự.
- [x] Lệnh tiền xử lý dữ liệu đã có sẵn.
- [ ] Lệnh huấn luyện đã có triển khai thật trong `src/train_lstm.py`.
- [ ] Lệnh đánh giá đã có triển khai thật trong `src/evaluate_lstm.py`.
- [ ] Lệnh dự đoán nhiệt độ ngày tiếp theo đã có triển khai thật trong `src/predict_next_day.py`.
- [x] Các file CLI không yêu cầu đường dẫn cục bộ tuyệt đối.
- [x] Không thêm chỉ số đánh giá giả.
- [x] Không thêm thao tác shuffle chuỗi thời gian.
- [x] Bước tiền xử lý lưu thông tin chia tập theo thứ tự thời gian trong `split_info.json`.
- [ ] File mô hình tồn tại tại `models/temp_lstm.keras`.
- [x] Thông báo khi thiếu mô hình đã rõ ràng.
- [ ] File chỉ số đánh giá tồn tại sau khi đánh giá thật.
- [ ] Biểu đồ tồn tại sau khi huấn luyện/đánh giá thật.
- [x] Thông báo khi thiếu chỉ số đánh giá/biểu đồ đã được ghi trong `test_log.txt`.
