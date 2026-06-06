# Tóm tắt dọn dẹp cấu trúc project

## Cập nhật sau khi hoàn thiện pipeline LSTM

- `src/windowing.py` đã được thay bằng logic tạo cửa sổ dữ liệu LSTM thật.
- `src/model_lstm.py` đã được thay bằng hàm build mô hình LSTM TensorFlow/Keras.
- `src/train_lstm.py` đã có pipeline huấn luyện, lưu model, history và biểu đồ.
- `src/evaluate_lstm.py` đã có pipeline đánh giá, đảo chuẩn hóa và tính MAE/MSE/RMSE.
- `src/predict_next_day.py` đã có pipeline dự đoán nhiệt độ ngày tiếp theo.
- `src/visualize_results.py` đã có các hàm lưu biểu đồ bằng Matplotlib.
- Các ghi chú cũ về "file giữ chỗ" chỉ còn giá trị lịch sử của lượt dọn dẹp trước.

## File đã giữ lại và lý do

- `README.md`: tài liệu chính hướng dẫn chạy project, cài thư viện, mô tả dữ liệu và checklist nghiệm thu.
- `requirements.txt`: file cài đặt thư viện chính của project.
- `main.py`: CLI chính hỗ trợ `self_test`, `preprocess`, `train`, `evaluate`, `predict_next_day`, `run_all`.
- `app_temperature_cli.py`: file bao lệnh đơn giản dùng lại logic từ `main.py`.
- `src/config.py`: cấu hình đường dẫn, tên cột, tham số chia tập và tham số huấn luyện.
- `src/data_utils.py`: có logic đọc, chuẩn hóa và kiểm tra dữ liệu nhiệt độ.
- `src/preprocess_timeseries.py`: có logic làm sạch, chia tập theo thời gian và chuẩn hóa dữ liệu.
- `src/check_processed_timeseries.py`: có logic kiểm tra các file dữ liệu đã xử lý.
- `src/windowing.py`, `src/model_lstm.py`, `src/train_lstm.py`, `src/evaluate_lstm.py`, `src/predict_next_day.py`, `src/visualize_results.py`: giữ lại vì đây là các module dự kiến của pipeline LSTM, dù một số file hiện vẫn là file giữ chỗ.
- `data/raw/temperature.csv`: dữ liệu đầu vào thật đang có trong project.
- `data/processed/*.csv`, `data/processed/split_info.json`: output tiền xử lý đang có, có thể dùng để kiểm tra pipeline dữ liệu.
- `models/scaler_params.json`: tham số chuẩn hóa đã được tạo từ bước tiền xử lý.
- `reports/01_data_profile/*.json`: hồ sơ dữ liệu được tạo từ bước tiền xử lý.
- `reports/final_report/submission_checklist.md`: checklist nộp bài cuối.
- `reports/final_report/test_log.txt`: log nghiệm thu và trạng thái lệnh kiểm tra.
- `tests/test_temperature_end_to_end.py`: script kiểm thử nghiệm thu nhẹ cho CLI và thông báo thiếu artifact.
- `tests/sample_temperature.csv`: dữ liệu mẫu nhỏ phục vụ kiểm thử, không dùng để báo cáo kết quả thật.
- Các file `.gitkeep`: giữ lại để Git theo dõi các thư mục cần có khi chưa phát sinh output thật.

## File đã gộp

- Không gộp file trong lượt dọn dẹp này.
- Các README nhỏ trong `data/raw/`, `data/processed/`, `models/`, `tests/` được giữ vì mô tả trực tiếp vai trò từng thư mục và không trùng hoàn toàn với `README.md`.

## File đã xóa và lý do

- `requirements-data.txt`: trùng một phần với `requirements.txt`, không được README, test, CLI hoặc source code tham chiếu.
- `__pycache__/`: cache Python sinh tự động, không cần nộp.
- `src/__pycache__/`: cache Python sinh tự động, không cần nộp.
- `tests/__pycache__/`: cache Python sinh tự động, không cần nộp.
- Các file `*.pyc`: bytecode Python sinh tự động, không cần nộp.

## File chưa xóa vì có thể vẫn cần

- `src/windowing.py`: hiện có dấu hiệu file giữ chỗ, nhưng là module cần thiết cho bước tạo cửa sổ chuỗi thời gian.
- `src/model_lstm.py`: hiện có dấu hiệu file giữ chỗ, nhưng là module cần thiết cho kiến trúc mô hình LSTM.
- `src/train_lstm.py`: hiện có dấu hiệu file giữ chỗ, nhưng là module cần thiết cho huấn luyện.
- `src/evaluate_lstm.py`: hiện có dấu hiệu file giữ chỗ, nhưng là module cần thiết cho đánh giá.
- `src/predict_next_day.py`: hiện có dấu hiệu file giữ chỗ, nhưng là module cần thiết cho dự đoán ngày tiếp theo.
- `src/visualize_results.py`: hiện có dấu hiệu file giữ chỗ, nhưng có thể cần cho biểu đồ huấn luyện/đánh giá.
- `models/.gitkeep`, `reports/02_training/.gitkeep`, `reports/03_evaluation/.gitkeep`: giữ lại vì model, metric và biểu đồ thật chưa có.

## Cần nhóm xác nhận trước khi xóa hoặc thay đổi

- Xác nhận có tiếp tục dùng các README nhỏ trong từng thư mục hay muốn gom toàn bộ vào `README.md`.
- Xác nhận có cần giữ `src/check_processed_timeseries.py` như công cụ kiểm tra riêng hay tích hợp hoàn toàn vào `main.py self_test`.
- Xác nhận thời điểm triển khai thật các module đang giữ chỗ: `windowing.py`, `model_lstm.py`, `train_lstm.py`, `evaluate_lstm.py`, `predict_next_day.py`, `visualize_results.py`.

## Cấu trúc đề xuất sau dọn dẹp

```text
temperature_forecast/
  README.md
  requirements.txt
  main.py
  app_temperature_cli.py
  data/
    raw/
      temperature.csv
      README_data.md
    processed/
      README_processed.md
      temperature_clean.csv
      temperature_train_scaled.csv
      temperature_val_scaled.csv
      temperature_test_scaled.csv
      temperature_scaled.csv
      split_info.json
  models/
    README_models.md
    scaler_params.json
  reports/
    01_data_profile/
    02_training/
    03_evaluation/
    final_report/
      submission_checklist.md
      test_log.txt
      cleanup_summary.md
  src/
    config.py
    data_utils.py
    preprocess_timeseries.py
    check_processed_timeseries.py
    windowing.py
    model_lstm.py
    train_lstm.py
    evaluate_lstm.py
    predict_next_day.py
    visualize_results.py
  tests/
    README_tests.md
    sample_temperature.csv
    test_temperature_end_to_end.py
```
