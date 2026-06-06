# Dữ liệu nhiệt độ gốc

Đặt file CSV nhiệt độ ban đầu trong thư mục này với đường dẫn:

```text
data/raw/temperature.csv
```

Các định dạng đầu vào được hỗ trợ:

- CSV ngày của NASA POWER có phần metadata ở đầu file và các cột `YEAR,DOY,T2M`.
- CSV ngày của NASA POWER có các cột `YEAR,MO,DY,T2M`.
- CSV thông thường có các cột `date,temperature`.

Quy tắc:

- Giữ nguyên file dữ liệu gốc, không chỉnh sửa trực tiếp.
- Giá trị thiếu của NASA được ký hiệu bằng `-999` và sẽ được xử lý như dữ liệu thiếu.
- File CSV gốc không được đưa lên Git; chỉ README này và `.gitkeep` được theo dõi.
