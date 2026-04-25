# README - Đồ án 2: Dự đoán nhiệt độ ngày tiếp theo

## 1. Tổng quan source code

Project `temperature_forecast` là source code cho đồ án dự đoán nhiệt độ ngày tiếp theo bằng mô hình LSTM.  
Source code được tổ chức theo hướng tách riêng dữ liệu, tiền xử lý chuỗi thời gian, tạo cửa sổ dữ liệu, xây dựng mô hình, huấn luyện, đánh giá, dự đoán và báo cáo.

Project tập trung vào các nhóm chức năng chính:

| Nhóm chức năng | Mô tả |
|---|---|
| Quản lý dữ liệu nhiệt độ | Lưu file CSV ban đầu và dữ liệu đã xử lý. |
| Làm sạch dữ liệu | Kiểm tra cột ngày, cột nhiệt độ, xử lý thiếu và sắp xếp theo thời gian. |
| Chuẩn hóa dữ liệu | Chuẩn hóa chuỗi nhiệt độ bằng thống kê từ tập train để tránh rò rỉ dữ liệu. |
| Tạo cửa sổ chuỗi thời gian | Biến chuỗi nhiệt độ thành dữ liệu đầu vào cho LSTM. |
| Xây dựng mô hình LSTM | Định nghĩa kiến trúc LSTM cho bài toán hồi quy một giá trị. |
| Huấn luyện mô hình | Huấn luyện model LSTM và lưu lịch sử huấn luyện. |
| Đánh giá mô hình | Tính MAE, MSE, RMSE và tạo biểu đồ thực tế so với dự đoán. |
| Dự đoán ngày tiếp theo | Lấy cửa sổ dữ liệu cuối cùng để dự đoán nhiệt độ ngày kế tiếp. |
| Báo cáo và nghiệm thu | Lưu hình ảnh, bảng kết quả, log kiểm thử và nội dung phục vụ báo cáo cuối. |

---

## 2. Cấu trúc thư mục tổng quát

| Đường dẫn | Chức năng |
|---|---|
| `temperature_forecast/` | Thư mục gốc của đồ án dự đoán nhiệt độ. Chứa toàn bộ source code, dữ liệu, model, báo cáo và file cấu hình project. |
| `temperature_forecast/data/` | Khu vực quản lý dữ liệu nhiệt độ. |
| `temperature_forecast/models/` | Khu vực lưu model LSTM và tham số chuẩn hóa. |
| `temperature_forecast/reports/` | Khu vực lưu kết quả trung gian và tài liệu phục vụ báo cáo. |
| `temperature_forecast/src/` | Khu vực chứa source code chính của project. |
| `temperature_forecast/tests/` | Khu vực chứa dữ liệu mẫu nhỏ và file phục vụ kiểm thử nhanh. |
| `temperature_forecast/app_temperature_cli.py` | File tích hợp giao diện dòng lệnh đơn giản cho project. |
| `temperature_forecast/main.py` | File điều phối chính để gọi các chức năng của project. |
| `temperature_forecast/requirements.txt` | File liệt kê thư viện cần thiết cho project. |
| `temperature_forecast/.gitignore` | File quy định những thành phần không đưa lên GitHub. |
| `temperature_forecast/README.md` | File mô tả cấu trúc source code, chức năng thư mục và chức năng file. |

---

## 3. Mô tả thư mục dữ liệu

| Đường dẫn | Chức năng |
|---|---|
| `data/raw/` | Chứa file CSV nhiệt độ ban đầu. File dữ liệu gốc không nên chỉnh sửa trực tiếp. |
| `data/raw/.gitkeep` | File giữ chỗ để GitHub lưu lại thư mục `raw` khi chưa có dữ liệu thật. |
| `data/raw/README_data.md` | Ghi chú định dạng dữ liệu nhiệt độ đầu vào. |
| `data/processed/` | Chứa dữ liệu đã làm sạch, dữ liệu đã chia train/validation/test hoặc dữ liệu đã chuẩn hóa. |
| `data/processed/.gitkeep` | File giữ chỗ để giữ thư mục `processed` trên GitHub. |
| `data/processed/README_processed.md` | Ghi chú vai trò của dữ liệu đã xử lý. |
| `data/processed/temperature_clean.csv` | File dữ liệu nhiệt độ sau khi làm sạch. File này chỉ xuất hiện sau khi chạy xử lý dữ liệu thật. |
| `data/processed/split_info.json` | File lưu thông tin chia tập train, validation và test. File này chỉ xuất hiện sau khi chạy xử lý dữ liệu thật. |

---

## 4. Mô tả thư mục model

| Đường dẫn | Chức năng |
|---|---|
| `models/` | Chứa model LSTM sau huấn luyện và các file liên quan đến chuẩn hóa dữ liệu. |
| `models/.gitkeep` | File giữ chỗ để giữ thư mục `models` trên GitHub. |
| `models/README_models.md` | Ghi chú vai trò của thư mục model. |
| `models/temp_lstm.keras` | File model LSTM sau khi huấn luyện. File này chỉ xuất hiện sau khi chạy huấn luyện thật. |
| `models/scaler_params.json` | File lưu tham số chuẩn hóa nhiệt độ, thường gồm mean và std tính từ tập train. |

---

## 5. Mô tả thư mục báo cáo

| Đường dẫn | Chức năng |
|---|---|
| `reports/` | Chứa toàn bộ kết quả trung gian và tài liệu phục vụ báo cáo đồ án. |
| `reports/01_data_profile/` | Lưu mô tả dữ liệu nhiệt độ, thống kê số dòng, khoảng thời gian, số giá trị thiếu và biểu đồ chuỗi nhiệt độ. |
| `reports/01_data_profile/.gitkeep` | File giữ chỗ cho thư mục báo cáo dữ liệu. |
| `reports/02_training/` | Lưu kết quả huấn luyện LSTM như lịch sử huấn luyện, biểu đồ loss, biểu đồ MAE và thông tin kiến trúc model. |
| `reports/02_training/.gitkeep` | File giữ chỗ cho thư mục kết quả huấn luyện. |
| `reports/03_evaluation/` | Lưu kết quả đánh giá LSTM như MAE, MSE, RMSE, bảng dự đoán và biểu đồ thực tế so với dự đoán. |
| `reports/03_evaluation/.gitkeep` | File giữ chỗ cho thư mục kết quả đánh giá. |
| `reports/final_report/` | Chứa hình ảnh, bảng kết quả và nội dung tổng hợp để đưa vào báo cáo cuối cùng. |
| `reports/final_report/.gitkeep` | File giữ chỗ cho thư mục báo cáo cuối. |

---

## 6. Mô tả thư mục source code

| File | Chức năng |
|---|---|
| `src/__init__.py` | Đánh dấu thư mục `src` là một package Python, hỗ trợ import module trong project. |
| `src/config.py` | Chứa các biến cấu hình dùng chung như đường dẫn dữ liệu, tên cột ngày, tên cột nhiệt độ, window size, split ratio, batch size, số epoch, seed và learning rate. |
| `src/data_utils.py` | Chứa các hàm đọc file CSV, kiểm tra dữ liệu đầu vào, kiểm tra cột bắt buộc và ghi thông tin tổng quan dữ liệu. |
| `src/preprocess_timeseries.py` | Chứa các hàm chuyển cột ngày sang dạng thời gian, sắp xếp dữ liệu, xử lý giá trị thiếu, chia tập theo thời gian và chuẩn hóa dữ liệu. |
| `src/windowing.py` | Chứa các hàm tạo cửa sổ chuỗi thời gian để biến dữ liệu nhiệt độ thành dữ liệu đầu vào phù hợp cho LSTM. |
| `src/model_lstm.py` | Chứa hàm xây dựng kiến trúc LSTM cho bài toán dự đoán nhiệt độ ngày tiếp theo. |
| `src/train_lstm.py` | Chứa quy trình huấn luyện LSTM, lưu model, lưu lịch sử huấn luyện và tạo kết quả phục vụ báo cáo huấn luyện. |
| `src/evaluate_lstm.py` | Chứa quy trình đánh giá model LSTM, tính MAE, MSE, RMSE và tạo bảng dự đoán. |
| `src/predict_next_day.py` | Chứa chức năng nạp model, lấy cửa sổ nhiệt độ cuối cùng và dự đoán nhiệt độ ngày tiếp theo. |
| `src/visualize_results.py` | Chứa các hàm vẽ biểu đồ chuỗi nhiệt độ, biểu đồ huấn luyện và biểu đồ thực tế so với dự đoán. |

---

## 7. Mô tả file tích hợp và kiểm thử

| File hoặc thư mục | Chức năng |
|---|---|
| `tests/` | Chứa dữ liệu mẫu nhỏ hoặc file hỗ trợ kiểm thử nhanh các module. |
| `tests/.gitkeep` | File giữ chỗ để giữ thư mục `tests` trên GitHub. |
| `tests/README_tests.md` | Mô tả vai trò của thư mục kiểm thử. |
| `tests/sample_temperature.csv` | File CSV mẫu nhỏ dùng để smoke test pipeline xử lý dữ liệu nhiệt độ. Không dùng file này để báo cáo kết quả thật. |
| `app_temperature_cli.py` | File xây dựng giao diện dòng lệnh đơn giản để gọi các chức năng train, evaluate hoặc predict next day. |
| `main.py` | File điều phối chính của project, dùng để gọi các pipeline hoặc kiểm thử tích hợp. |
| `requirements.txt` | File quản lý danh sách thư viện cần cài đặt. |
| `.gitignore` | File loại trừ cache, môi trường ảo, dữ liệu lớn, model lớn hoặc file tạm khỏi GitHub. |
| `README.md` | File giới thiệu source code và mô tả chức năng từng thư mục, từng file. |

---

## 8. Các file kết quả sẽ phát sinh khi chạy project

| File kết quả | Thư mục lưu | Chức năng |
|---|---|---|
| `temperature_clean.csv` | `data/processed/` | Dữ liệu nhiệt độ sau khi làm sạch. |
| `split_info.json` | `data/processed/` | Thông tin chia tập train, validation và test. |
| `scaler_params.json` | `models/` | Tham số chuẩn hóa được tính từ tập train. |
| `temp_lstm.keras` | `models/` | Model LSTM sau khi huấn luyện. |
| `data_profile.csv` hoặc `data_profile.txt` | `reports/01_data_profile/` | Thống kê tổng quan dữ liệu nhiệt độ. |
| `temperature_series.png` | `reports/01_data_profile/` | Biểu đồ chuỗi nhiệt độ theo thời gian. |
| `window_config.json` | `reports/02_training/` | Thông tin cấu hình cửa sổ dữ liệu. |
| `lstm_model_summary.txt` | `reports/02_training/` | Tóm tắt kiến trúc model LSTM. |
| `lstm_history.csv` | `reports/02_training/` | Lịch sử huấn luyện qua từng epoch. |
| `lstm_loss_curve.png` | `reports/02_training/` | Biểu đồ loss hoặc MAE trong quá trình huấn luyện. |
| `regression_metrics.json` | `reports/03_evaluation/` | Kết quả MAE, MSE và RMSE trên tập đánh giá. |
| `predictions.csv` | `reports/03_evaluation/` | Bảng ngày, nhiệt độ thật, nhiệt độ dự đoán và sai số. |
| `actual_vs_predicted.png` | `reports/03_evaluation/` | Biểu đồ so sánh nhiệt độ thật và nhiệt độ dự đoán. |
| `next_day_prediction.txt` | `reports/03_evaluation/` | Kết quả dự đoán nhiệt độ ngày tiếp theo. |

---

## 9. Quy ước quản lý source code

| Quy ước | Mô tả |
|---|---|
| Tách module rõ ràng | Mỗi file trong `src/` phụ trách một nhóm chức năng riêng để dễ chia việc cho thành viên. |
| Không shuffle chuỗi thời gian | Dữ liệu nhiệt độ phải được chia theo thứ tự thời gian để tránh rò rỉ dữ liệu. |
| Không fit chuẩn hóa trên toàn bộ dữ liệu | Tham số chuẩn hóa chỉ được tính từ tập train, sau đó áp dụng cho validation và test. |
| Không ghi kết quả ảo | Các chỉ số MAE, MSE, RMSE chỉ được ghi sau khi chạy thật. |
| Không đẩy dữ liệu lớn nếu không cần | Dữ liệu thật và model lớn có thể được loại khỏi GitHub bằng `.gitignore`. |
| Giữ thư mục rỗng bằng `.gitkeep` | Các thư mục chưa có dữ liệu vẫn được lưu trong GitHub nhờ file `.gitkeep`. |
| Báo cáo lấy từ output thật | Hình ảnh và bảng kết quả trong báo cáo phải lấy từ thư mục `reports/`. |