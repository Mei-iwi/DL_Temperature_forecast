# Dự đoán nhiệt độ ngày tiếp theo bằng LSTM

## Mục tiêu đề tài

Dự án `temperature_forecast` thuộc chủ đề 2 môn Thực hành Deep Learning. Mục tiêu là xây dựng quy trình dự đoán nhiệt độ ngày tiếp theo từ chuỗi nhiệt độ theo thời gian bằng mô hình LSTM với TensorFlow/Keras.

Dự án tập trung vào các bước:

- Đọc dữ liệu nhiệt độ từ file CSV.
- Làm sạch dữ liệu, chuẩn hóa cột ngày và cột nhiệt độ.
- Chia dữ liệu theo thứ tự thời gian thành train, validation và test.
- Chuẩn hóa nhiệt độ bằng tham số tính từ tập train để tránh rò rỉ dữ liệu.
- Huấn luyện mô hình LSTM khi mô-đun huấn luyện đã được hoàn thiện.
- Đánh giá bằng MAE, MSE, RMSE và lưu kết quả thật sau khi chạy.
- Dự đoán nhiệt độ ngày tiếp theo từ cửa sổ dữ liệu gần nhất.

## Công nghệ sử dụng

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- TensorFlow/Keras
- pathlib, argparse, json và các thư viện chuẩn của Python

Không dùng Flask, FastAPI, Streamlit, AutoML hoặc framework web không liên quan.

## Cấu trúc thư mục

```text
temperature_forecast/
  data/
    raw/
      temperature.csv
    processed/
  models/
  reports/
    01_data_profile/
    02_training/
    03_evaluation/
    final_report/
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
  tests/
    test_temperature_end_to_end.py
  main.py
  app_temperature_cli.py
  requirements.txt
```

## Dữ liệu đầu vào

Đặt dữ liệu gốc tại:

```text
data/raw/temperature.csv
```

File CSV cần có một trong các dạng sau:

- Cột `date` và `temperature`.
- Dữ liệu NASA POWER có `YEAR`, `MO`, `DY`, `T2M`.
- Dữ liệu NASA POWER có `YEAR`, `DOY`, `T2M`.

Chuỗi thời gian phải được xử lý theo thứ tự thời gian. Không shuffle dữ liệu khi chia tập.

## Tạo môi trường ảo

Trên Windows PowerShell hoặc Command Prompt:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Nếu đang dùng Git Bash/MINGW64 trên Windows, dùng đường dẫn `Scripts`:

```bash
py -m venv .venv
source .venv/Scripts/activate
```

Trên macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Lưu ý: trong Git Bash trên Windows, lệnh `source .venv/bin/activate` thường báo lỗi `No such file or directory` vì môi trường ảo Windows tạo file kích hoạt trong `.venv/Scripts/activate`.

## Cài thư viện

```bash
pip install -r requirements.txt
```

## Cách chạy project

Kiểm tra cấu trúc dự án, import mô-đun và các file kết quả hiện có:

```bash
python main.py self_test
```

Tiền xử lý dữ liệu:

```bash
python main.py preprocess
```

Huấn luyện mô hình:

```bash
python main.py train
```

Đánh giá mô hình:

```bash
python main.py evaluate
```

Dự đoán nhiệt độ ngày tiếp theo:

```bash
python main.py predict_next_day
```

Chạy toàn bộ quy trình theo thứ tự:

```bash
python main.py run_all
```

Có thể dùng file bao lệnh đơn giản:

```bash
python app_temperature_cli.py self_test
python app_temperature_cli.py preprocess
python app_temperature_cli.py train
python app_temperature_cli.py evaluate
python app_temperature_cli.py predict_next_day
python app_temperature_cli.py run_all
```

## Vị trí kết quả

Các file đã có hoặc sẽ được tạo sau khi chạy đúng quy trình:

| Kết quả | Vị trí |
|---|---|
| Dữ liệu sạch | `data/processed/temperature_clean.csv` |
| Dữ liệu train đã chuẩn hóa | `data/processed/temperature_train_scaled.csv` |
| Dữ liệu validation đã chuẩn hóa | `data/processed/temperature_val_scaled.csv` |
| Dữ liệu test đã chuẩn hóa | `data/processed/temperature_test_scaled.csv` |
| Thông tin chia tập | `data/processed/split_info.json` |
| Tham số chuẩn hóa | `models/scaler_params.json` |
| Mô hình LSTM | `models/temp_lstm.keras` |
| Hồ sơ dữ liệu | `reports/01_data_profile/` |
| Lịch sử huấn luyện | `reports/02_training/` |
| Chỉ số đánh giá và dự đoán | `reports/03_evaluation/` |
| Log nghiệm thu | `reports/final_report/test_log.txt` |
| Checklist nộp bài | `reports/final_report/submission_checklist.md` |

Nếu nhóm đổi tên file kết quả trong `src/config.py`, hãy ưu tiên đường dẫn đang được cấu hình trong file đó.

## Giải thích chỉ số đánh giá

- MAE: Sai số tuyệt đối trung bình giữa nhiệt độ thật và nhiệt độ dự đoán.
- MSE: Trung bình bình phương sai số, phạt nặng hơn các lỗi lớn.
- RMSE: Căn bậc hai của MSE, có cùng đơn vị với nhiệt độ nên dễ diễn giải hơn.

Chỉ ghi MAE, MSE, RMSE vào báo cáo sau khi chạy đánh giá thật. Không tự điền số liệu.

## Checklist nghiệm thu trước khi nộp

- `README.md` đã hướng dẫn đầy đủ cách chạy.
- `requirements.txt` chỉ chứa thư viện cần thiết.
- `main.py` hỗ trợ `self_test`, `preprocess`, `train`, `evaluate`, `predict_next_day`, `run_all`.
- `app_temperature_cli.py` gọi được các lệnh tương tự.
- Dữ liệu gốc nằm tại `data/raw/temperature.csv`.
- Dữ liệu được chia theo thời gian, không shuffle.
- Chuẩn hóa chỉ tính tham số trên tập train.
- Không dùng dữ liệu tương lai trong huấn luyện.
- Không ghi chỉ số đánh giá hoặc kết quả dự đoán giả.
- Mô hình, biểu đồ, chỉ số đánh giá và kết quả dự đoán trong báo cáo phải lấy từ file kết quả thật.

## Ghi chú bắt buộc

Ở trạng thái hiện tại, một số mô-đun như `train_lstm.py`, `evaluate_lstm.py`, `predict_next_day.py`, `model_lstm.py` và `windowing.py` vẫn có dấu hiệu là file giữ chỗ. CLI sẽ báo rõ khi chức năng chưa được triển khai thay vì tạo kết quả giả.
