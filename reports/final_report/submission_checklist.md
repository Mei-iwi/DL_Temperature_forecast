# Checklist nghiệm thu nộp bài

- [x] `requirements.txt` đã có đủ thư viện cần thiết.
- [x] `README.md` đã cập nhật hướng dẫn tiếng Việt.
- [x] `main.py` hỗ trợ `self_test`, `preprocess`, `train`, `evaluate`, `predict_next_day`, `run_all`.
- [x] `app_temperature_cli.py` gọi lại được các lệnh CLI chính.
- [x] `src/windowing.py` đã có hàm tạo cửa sổ dữ liệu LSTM.
- [x] `src/model_lstm.py` đã có hàm build mô hình LSTM.
- [x] `src/train_lstm.py` đã có pipeline huấn luyện và lưu model/history/biểu đồ.
- [x] `src/evaluate_lstm.py` đã có pipeline đánh giá, đảo chuẩn hóa và tính MAE/MSE/RMSE.
- [x] `src/predict_next_day.py` đã có pipeline dự đoán ngày tiếp theo.
- [x] `src/visualize_results.py` đã có các hàm vẽ biểu đồ.
- [x] `tests/test_temperature_end_to_end.py` đã kiểm tra CLI, windowing, metric và input dự đoán bằng dữ liệu giả.
- [x] Không thêm metric giả hoặc kết quả dự đoán giả.
- [x] Dữ liệu được chia theo thời gian, không shuffle.
- [x] Scaler chỉ fit trên train.
- [x] Lệnh `python main.py run_all` đã được ghi trong README.
- [ ] Chạy train/evaluate/predict thật trong môi trường đã cài đủ TensorFlow/Pandas.
- [ ] Xác nhận model thật tồn tại tại `models/temperature_lstm.keras` sau khi train.
- [ ] Xác nhận metric thật tồn tại tại `outputs/metrics/regression_metrics.csv` sau khi evaluate.
