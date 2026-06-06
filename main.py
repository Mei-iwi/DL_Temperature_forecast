from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable

from src import config


PROJECT_ROOT = Path(__file__).resolve().parent

REQUIRED_DIRS = (
    "src",
    "data",
    "data/raw",
    "data/processed",
    "models",
    "reports",
    "reports/01_data_profile",
    "reports/02_training",
    "reports/03_evaluation",
    "reports/final_report",
    "outputs",
    "outputs/figures",
    "outputs/metrics",
    "outputs/logs",
    "tests",
)

REQUIRED_FILES = (
    "README.md",
    "requirements.txt",
    "src/config.py",
    "src/data_utils.py",
    "src/preprocess_timeseries.py",
    "src/train_lstm.py",
    "src/evaluate_lstm.py",
    "src/predict_next_day.py",
    "src/model_lstm.py",
    "src/windowing.py",
)

MODULES_TO_CHECK = (
    "src.config",
    "src.data_utils",
    "src.preprocess_timeseries",
    "src.train_lstm",
    "src.evaluate_lstm",
    "src.predict_next_day",
    "src.model_lstm",
    "src.windowing",
)


class CliError(RuntimeError):
    """Lỗi CLI có thông báo rõ ràng cho người dùng.

    Hàm CLI sẽ bắt lỗi này và in hướng dẫn thay vì hiển thị traceback khó đọc.
    """


def rel_path(path: str | Path) -> str:
    """Chuyển đường dẫn về dạng tương đối so với thư mục project.

    Tham số `path` là đường dẫn file hoặc thư mục cần hiển thị.
    Trả về chuỗi đường dẫn ngắn gọn để thông báo CLI không phụ thuộc máy cá nhân.
    """
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def safe_import(module_name: str) -> ModuleType | None:
    """Import module một cách an toàn để phục vụ self_test.

    Tham số `module_name` là tên module Python cần kiểm tra.
    Trả về module nếu import được, ngược lại in lỗi dễ hiểu và trả về None.
    """
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - message path depends on local environment
        print(f"[MISSING] Cannot import {module_name}: {exc}")
        return None


def find_callable(module_name: str, candidates: tuple[str, ...]) -> Callable[..., object]:
    """Tìm hàm triển khai thật trong một module.

    `module_name` là module cần kiểm tra, `candidates` là các tên hàm hợp lệ.
    Trả về hàm đầu tiên tồn tại; nếu chưa có thì báo lỗi rõ để nhóm bổ sung.
    """
    module = safe_import(module_name)
    if module is None:
        raise CliError(f"Module {module_name} is missing or has import errors.")

    for function_name in candidates:
        function = getattr(module, function_name, None)
        if callable(function):
            return function

    names = ", ".join(candidates)
    raise CliError(
        f"Module {module_name} does not expose an implementation function. "
        f"Expected one of: {names}."
    )


def ensure_raw_data_exists() -> None:
    """Kiểm tra file dữ liệu gốc trước khi tiền xử lý.

    Hàm yêu cầu `data/raw/temperature.csv` tồn tại.
    Nếu thiếu, CLI sẽ hướng dẫn đặt dữ liệu đúng vị trí thay vì bị crash.
    """
    if not config.DATA_RAW_PATH.exists():
        raise CliError(
            f"Missing input data: {rel_path(config.DATA_RAW_PATH)}. "
            "Please place the raw CSV at data/raw/temperature.csv, then run: python main.py preprocess"
        )


def ensure_processed_data_exists() -> None:
    """Kiểm tra các file dữ liệu đã xử lý trước khi train/evaluate/predict.

    Hàm đảm bảo output từ bước `preprocess` đã có đủ.
    Lưu ý: dữ liệu đã xử lý phải được chia theo thời gian, không shuffle.
    """
    required_groups = (
        (config.CLEAN_DATA_PATH,),
        (config.TRAIN_CSV_PATH, config.TRAIN_SCALED_PATH),
        (config.VAL_CSV_PATH, config.VAL_SCALED_PATH),
        (config.TEST_CSV_PATH, config.TEST_SCALED_PATH),
        (config.SPLIT_INFO_PATH,),
        (config.SCALER_PARAMS_PATH,),
    )
    missing = [
        " hoặc ".join(rel_path(path) for path in group)
        for group in required_groups
        if not any(Path(path).exists() for path in group)
    ]
    if missing:
        raise CliError(
            "Missing processed data files: "
            + ", ".join(missing)
            + ". Please run first: python main.py preprocess"
        )


def ensure_model_exists() -> None:
    """Kiểm tra model LSTM trước khi đánh giá hoặc dự đoán.

    Hàm yêu cầu file model tại đường dẫn cấu hình trong `config.MODEL_PATH`.
    Nếu thiếu model, người dùng cần chạy huấn luyện thật trước.
    """
    if not config.MODEL_PATH.exists():
        raise CliError(
            f"Missing trained model: {rel_path(config.MODEL_PATH)}. "
            "Please run training first after the training module is implemented: python main.py train"
        )


def check_placeholder_module(module_name: str) -> str:
    """Nhận diện nhanh module còn là file giữ chỗ.

    Tham số `module_name` là tên module cần kiểm tra.
    Trả về `missing`, `placeholder` hoặc `present` để self_test báo trạng thái.
    """
    module_path = PROJECT_ROOT / Path(module_name.replace(".", "/") + ".py")
    if not module_path.exists():
        return "missing"
    text = module_path.read_text(encoding="utf-8")
    if "placeholder" in text.lower():
        return "placeholder"
    return "present"


def self_test() -> int:
    """Chạy kiểm tra tích hợp nhẹ cho project.

    Hàm kiểm tra thư mục, file bắt buộc, import module và các artifact quan trọng.
    Trả về mã 0 nếu self_test hoàn tất, kể cả khi có cảnh báo thiếu data/model.
    """
    print("[INFO] Running temperature_forecast self-test")

    # Kiểm tra cấu trúc thư mục tối thiểu cần có để nộp project.
    for directory in REQUIRED_DIRS:
        path = PROJECT_ROOT / directory
        if path.exists():
            print(f"[OK] folder: {directory}")
        else:
            print(f"[MISSING] folder: {directory}")

    # Kiểm tra các file nguồn và tài liệu quan trọng.
    for file_name in REQUIRED_FILES:
        path = PROJECT_ROOT / file_name
        if path.exists():
            print(f"[OK] file: {file_name}")
        else:
            print(f"[MISSING] file: {file_name}")

    # Import thử từng module để phát hiện lỗi phụ thuộc hoặc file giữ chỗ.
    for module_name in MODULES_TO_CHECK:
        module = safe_import(module_name)
        if module is not None:
            status = check_placeholder_module(module_name)
            print(f"[OK] import: {module_name} ({status})")

    print(f"[INFO] raw data path: {rel_path(config.DATA_RAW_PATH)}")
    print(f"[INFO] model path: {rel_path(config.MODEL_PATH)}")
    print(f"[INFO] window size: {config.WINDOW_SIZE}")
    print(f"[INFO] train/validation/test split: {config.TRAIN_RATIO}/{config.VAL_RATIO}/{1 - config.TRAIN_RATIO - config.VAL_RATIO:.2f}")

    if not config.DATA_RAW_PATH.exists():
        print("[NOTICE] Raw data is missing. Add data/raw/temperature.csv before preprocessing.")
    if not config.MODEL_PATH.exists():
        print("[NOTICE] Model artifact is missing. Run training after train_lstm.py is implemented.")

    print("[OK] self_test completed")
    return 0


def preprocess() -> int:
    """Chạy bước tiền xử lý dữ liệu nhiệt độ.

    Đầu vào chính là `data/raw/temperature.csv`.
    Output gồm dữ liệu sạch, dữ liệu train/validation/test đã chuẩn hóa và scaler_params.json.
    """
    ensure_raw_data_exists()
    function = find_callable(
        "src.preprocess_timeseries",
        ("preprocess_temperature_pipeline", "preprocess_timeseries", "main"),
    )
    result = function()
    if isinstance(result, dict) and "clean_profile" in result:
        profile = result["clean_profile"]
        print(f"[OK] preprocess completed: {profile.get('rows')} clean rows")
    else:
        print("[OK] preprocess completed")
    return 0


def train() -> int:
    """Chạy bước huấn luyện mô hình LSTM.

    Hàm yêu cầu dữ liệu đã xử lý từ bước `preprocess`.
    Model và history chỉ được tạo bởi module huấn luyện thật, không tạo kết quả giả.
    """
    ensure_processed_data_exists()
    function = find_callable(
        "src.train_lstm",
        ("train_lstm_pipeline", "train_lstm", "train_temperature_lstm", "train_model", "train"),
    )
    function()
    print("[OK] train completed")
    return 0


def evaluate() -> int:
    """Chạy bước đánh giá mô hình LSTM.

    Hàm yêu cầu dữ liệu đã xử lý và model đã huấn luyện.
    Các chỉ số MAE, MSE, RMSE phải đến từ module đánh giá thật.
    """
    ensure_processed_data_exists()
    ensure_model_exists()
    function = find_callable(
        "src.evaluate_lstm",
        ("evaluate_lstm_pipeline", "evaluate_lstm", "evaluate_temperature_lstm", "evaluate_model", "evaluate"),
    )
    function()
    print("[OK] evaluate completed")
    return 0


def predict_next_day() -> int:
    """Chạy bước dự đoán nhiệt độ ngày tiếp theo.

    Hàm yêu cầu dữ liệu đã xử lý và model đã huấn luyện.
    Kết quả dự đoán phải được tính từ model thật, không ghi số liệu giả.
    """
    ensure_processed_data_exists()
    ensure_model_exists()
    function = find_callable(
        "src.predict_next_day",
        ("predict_next_day", "predict_next_temperature", "run_prediction", "predict"),
    )
    function()
    print("[OK] predict_next_day completed")
    return 0


def run_all() -> int:
    """Chạy toàn bộ quy trình theo thứ tự chuẩn.

    Thứ tự là preprocess -> train -> evaluate -> predict_next_day.
    Nếu một bước thiếu dữ liệu, model hoặc hàm triển khai, CLI sẽ dừng với thông báo rõ.
    """
    steps = (preprocess, train, evaluate, predict_next_day)
    for step in steps:
        print(f"[INFO] Running step: {step.__name__}")
        step()
    print("[OK] run_all completed")
    return 0


COMMANDS: dict[str, Callable[[], int]] = {
    "self_test": self_test,
    "preprocess": preprocess,
    "train": train,
    "evaluate": evaluate,
    "predict_next_day": predict_next_day,
    "run_all": run_all,
}


def build_parser() -> argparse.ArgumentParser:
    """Tạo bộ phân tích tham số dòng lệnh.

    Hàm định nghĩa các lệnh CLI hợp lệ và giữ nguyên cú pháp như README hướng dẫn.
    Trả về đối tượng argparse.ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="CLI for the LSTM next-day temperature forecasting project."
    )
    parser.add_argument(
        "command",
        choices=sorted(COMMANDS),
        help="Command to run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Điểm vào chính của CLI.

    `argv` là danh sách tham số dòng lệnh, dùng để test hoặc chạy thật.
    Trả về mã trạng thái: 0 khi thành công, 1/2 khi có lỗi được xử lý.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return COMMANDS[args.command]()
    except CliError as exc:
        print(f"[ERROR] {exc}")
        return 2
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 2
    except Exception as exc:
        print(f"[ERROR] {args.command} failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
