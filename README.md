# Dự đoán nhiệt độ ngày tiếp theo bằng LSTM

## 1. Mục tiêu đề tài

Project `temperature_forecast` xây dựng một pipeline dự đoán nhiệt độ ngày tiếp theo từ dữ liệu nhiệt độ dạng chuỗi thời gian.

Các mục tiêu chính:

- Đọc dữ liệu nhiệt độ từ file CSV.
- Tiền xử lý dữ liệu: chuẩn hóa ngày tháng, xử lý giá trị thiếu, sắp xếp theo thời gian.
- Chia train/validation/test theo thứ tự thời gian, không shuffle.
- Chuẩn hóa nhiệt độ bằng thống kê fit từ tập train.
- Tạo cửa sổ dữ liệu đầu vào cho mô hình LSTM.
- Huấn luyện mô hình LSTM bằng TensorFlow/Keras.
- Đánh giá bằng MAE, MSE, RMSE.
- Dự đoán nhiệt độ ngày tiếp theo từ `WINDOW_SIZE` ngày gần nhất.

## 2. Công nghệ sử dụng

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow/Keras
- pathlib, argparse, json và các thư viện chuẩn của Python

Không dùng Flask, FastAPI, Streamlit, AutoML, CNN hoặc nội dung phân loại ảnh.

## 3. Cấu trúc thư mục

```text
temperature_forecast/
  README.md
  requirements.txt
  main.py
  app_temperature_cli.py
  docs/
    architecture_source_for_chatgpt.md
  data/
    raw/
      temperature.csv
    processed/
      temperature_clean.csv
      train_series.csv
      val_series.csv
      test_series.csv
      split_info.json
  src/
    config.py
    data_utils.py
    preprocess_timeseries.py
    windowing.py
    model_lstm.py
    train_lstm.py
    evaluate_lstm.py
    predict_next_day.py
    visualize_results.py
  models/
    temp_lstm.keras
    scaler_params.json
  outputs/
    figures/
    metrics/
    logs/
  reports/
    final_report/
  tests/
    test_temperature_end_to_end.py
```

Vai trò chính:

| File/thư mục | Chức năng |
|---|---|
| `main.py` | CLI chính để chạy `self_test`, `preprocess`, `train`, `evaluate`, `predict_next_day`, `run_all`. |
| `app_temperature_cli.py` | Wrapper đơn giản dùng lại CLI trong `main.py`. |
| `src/config.py` | Khai báo đường dẫn, tên cột, tham số LSTM và output. |
| `src/data_utils.py` | Đọc CSV, chuẩn hóa schema dữ liệu, kiểm tra dữ liệu đầu vào. |
| `src/preprocess_timeseries.py` | Làm sạch, chia tập theo thời gian, chuẩn hóa và lưu dữ liệu processed. |
| `src/windowing.py` | Tạo cửa sổ dữ liệu LSTM từ chuỗi nhiệt độ đã chuẩn hóa. |
| `src/model_lstm.py` | Xây dựng mô hình LSTM hồi quy một giá trị. |
| `src/train_lstm.py` | Huấn luyện LSTM, lưu model, history và biểu đồ huấn luyện. |
| `src/evaluate_lstm.py` | Đánh giá model trên test set, đảo chuẩn hóa, tính MAE/MSE/RMSE. |
| `src/predict_next_day.py` | Dự đoán nhiệt độ ngày tiếp theo từ cửa sổ mới nhất. |
| `src/visualize_results.py` | Vẽ biểu đồ chuỗi nhiệt độ, lịch sử train và kết quả dự đoán. |
| `tests/test_temperature_end_to_end.py` | Kiểm thử nhẹ, không train model thật. |

Tài liệu kiến trúc chi tiết nằm tại:

```text
docs/architecture_source_for_chatgpt.md
```

File này mô tả các lớp pipeline, luồng dữ liệu, kiến trúc LSTM, artifact đầu ra và Mermaid diagram để có thể đưa sang ChatGPT Web tạo phần giải thích kiến trúc hoặc hình minh họa chi tiết.

## 4. Chuẩn bị dữ liệu

Đặt file CSV tại:

```text
data/raw/temperature.csv
```

Các dạng cột được hỗ trợ:

- CSV thường: `date`, `temperature`
- NASA POWER: `YEAR`, `MO`, `DY`, `T2M`
- NASA POWER: `YEAR`, `DOY`, `T2M`

Trong NASA POWER, `T2M` là nhiệt độ trung bình gần mặt đất. Sau khi đọc, project chuẩn hóa về hai cột chính: `date` và `temperature`.

## 5. Cài đặt môi trường

Tạo môi trường ảo:

```bash
python -m venv .venv
```

Windows PowerShell hoặc Command Prompt:

```bash
.venv\Scripts\activate
```

Git Bash/MINGW64 trên Windows:

```bash
py -m venv .venv
source .venv/Scripts/activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Cài thư viện:

```bash
pip install -r requirements.txt
```

Lưu ý: nếu dùng Git Bash trên Windows, `source .venv/bin/activate` thường lỗi vì môi trường ảo Windows đặt file kích hoạt trong `.venv/Scripts/activate`.

## 6. Kiểm thử nhanh

```bash
python main.py self_test
python tests/test_temperature_end_to_end.py
python app_temperature_cli.py self_test
```

Test nhẹ chỉ kiểm tra cấu trúc, CLI, tạo window, tính metric giả lập và reshape input dự đoán. Test không train model thật và không cần model đã lưu.

## 7. Chạy từng bước

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

## 8. Chạy toàn bộ một lần

```bash
python main.py run_all
```

Lệnh này chạy theo thứ tự:

```text
preprocess -> train -> evaluate -> predict_next_day
```

Nếu thiếu dữ liệu, thiếu thư viện hoặc thiếu model, CLI sẽ in thông báo rõ ràng. Không tự ghi metric hoặc kết quả dự đoán giả.

## 9. Kết quả đầu ra

Sau khi chạy thật, project tạo các output chính:

| Kết quả | Đường dẫn |
|---|---|
| Dữ liệu sạch | `data/processed/temperature_clean.csv` |
| Tập train | `data/processed/train_series.csv` |
| Tập validation | `data/processed/val_series.csv` |
| Tập test | `data/processed/test_series.csv` |
| Thông tin chia tập | `data/processed/split_info.json` |
| Tham số chuẩn hóa | `models/scaler_params.json` |
| Model LSTM | `models/temp_lstm.keras` |
| Lịch sử huấn luyện | `reports/02_training/lstm_history.csv` |
| Biểu đồ chuỗi nhiệt độ | `outputs/figures/temperature_series.png` |
| Biểu đồ loss/MAE | `reports/02_training/lstm_loss_curve.png` |
| Biểu đồ thực tế/dự đoán | `outputs/figures/actual_vs_predicted.png` |
| Biểu đồ residual | `outputs/figures/residual_plot.png` |
| Chỉ số đánh giá | `outputs/metrics/regression_metrics.csv` |
| Bảng dự đoán test | `outputs/metrics/predictions_test.csv` |
| Dự đoán ngày tiếp theo | `outputs/metrics/next_day_prediction.txt` |

## 10. Ý nghĩa chỉ số đánh giá

- MAE: Sai số tuyệt đối trung bình giữa nhiệt độ thật và nhiệt độ dự đoán.
- MSE: Trung bình bình phương sai số, phạt mạnh hơn các lỗi lớn.
- RMSE: Căn bậc hai của MSE, cùng đơn vị với nhiệt độ nên dễ diễn giải.

Chỉ đưa các chỉ số này vào báo cáo sau khi chạy `python main.py evaluate` thật.

## 11. Lưu ý chống rò rỉ dữ liệu

- Không shuffle dữ liệu chuỗi thời gian.
- Chia train/validation/test theo thứ tự thời gian: train trước, validation sau, test cuối.
- Chỉ fit scaler trên tập train.
- Validation/test dùng lại mean/std từ train.
- Dự đoán ngày tiếp theo chỉ dùng `WINDOW_SIZE` ngày gần nhất.
- Không dùng dữ liệu tương lai để tạo nhãn huấn luyện hoặc dự đoán.

## 12. Phân công thành viên

- Nguyễn Phi Hổ: tạo cửa sổ dữ liệu, xây dựng LSTM, huấn luyện mô hình.
- Dư Vũ Khang: đánh giá LSTM, đảo chuẩn hóa, dự đoán nhiệt độ ngày tiếp theo.

## 13. Checklist nghiệm thu

- [ ] Có dữ liệu raw tại `data/raw/temperature.csv`.
- [ ] `python main.py preprocess` chạy được.
- [ ] Windowing tạo đúng shape `(samples, WINDOW_SIZE, 1)`.
- [ ] Model LSTM build được.
- [ ] Train tạo `models/temp_lstm.keras`.
- [ ] Evaluate tạo MAE, MSE, RMSE thật.
- [ ] Predict tạo kết quả ngày tiếp theo.
- [ ] README có hướng dẫn cài đặt, chạy test và chạy project.
- [ ] Không có số liệu giả.
- [ ] Không shuffle chuỗi thời gian.
- [ ] Không fit scaler trên validation/test.
