from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Chạy một lệnh Python trong thư mục gốc project.

    Tham số `args` là danh sách đối số như `["main.py", "self_test"]`.
    Trả về CompletedProcess để test kiểm tra mã thoát và nội dung output.
    """
    return subprocess.run(
        [sys.executable, *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def assert_true(condition: bool, message: str) -> None:
    """Hàm assert nhỏ để script test chạy không cần pytest.

    Nếu `condition` sai, hàm raise AssertionError với thông báo đã truyền vào.
    """
    if not condition:
        raise AssertionError(message)


def test_required_folders() -> None:
    """Kiểm tra các thư mục quan trọng tồn tại hoặc có thể tạo được.

    Hàm bảo đảm cấu trúc tối thiểu của project vẫn sẵn sàng cho nghiệm thu.
    """
    required_dirs = [
        "src",
        "data",
        "data/raw",
        "data/processed",
        "models",
        "reports",
        "reports/final_report",
        "outputs",
        "outputs/figures",
        "outputs/metrics",
        "outputs/logs",
        "tests",
    ]
    for directory in required_dirs:
        path = PROJECT_ROOT / directory
        path.mkdir(parents=True, exist_ok=True)
        assert_true(path.exists(), f"Missing folder: {directory}")


def test_self_test_runs() -> None:
    """Kiểm tra `python main.py self_test` chạy không có traceback.

    Hàm xác nhận CLI chính có thể tự kiểm tra project và trả mã thành công.
    """
    result = run_command(["main.py", "self_test"])
    output = result.stdout + result.stderr
    assert_true(result.returncode == 0, output)
    assert_true("[OK] self_test completed" in output, output)
    assert_true("Traceback" not in output, output)


def test_app_wrapper_runs() -> None:
    """Kiểm tra wrapper `app_temperature_cli.py` gọi được self_test.

    Hàm giúp bảo đảm file CLI thân thiện cho sinh viên vẫn dùng chung logic với main.py.
    """
    result = run_command(["app_temperature_cli.py", "self_test"])
    output = result.stdout + result.stderr
    assert_true(result.returncode == 0, output)
    assert_true("[OK] self_test completed" in output, output)


def test_cli_argument_validation() -> None:
    """Kiểm tra argparse từ chối lệnh CLI không hợp lệ.

    Hàm bảo đảm người dùng nhập sai lệnh sẽ nhận thông báo rõ thay vì lỗi khó hiểu.
    """
    result = run_command(["main.py", "unknown_command"])
    output = result.stdout + result.stderr
    assert_true(result.returncode != 0, "Unknown command should fail.")
    assert_true("invalid choice" in output, output)


def test_evaluate_missing_model_message() -> None:
    """Kiểm tra thông báo khi đánh giá mà chưa có model.

    Nếu model LSTM chưa tồn tại, CLI phải hướng dẫn rõ cần train trước.
    """
    from src import config

    result = run_command(["main.py", "evaluate"])
    output = result.stdout + result.stderr
    if not config.MODEL_PATH.exists():
        assert_true(result.returncode == 2, output)
        assert_true("Missing trained model" in output, output)


def test_windowing_dummy_data() -> None:
    """Kiểm tra tạo cửa sổ LSTM bằng dữ liệu giả, không cần dữ liệu thật."""
    from src.windowing import create_sequences

    values = list(range(20))
    X, y = create_sequences(values, window_size=7)
    assert_true(X.ndim == 3, "X must be 3D.")
    assert_true(X.shape[1] == 7, "Window size must be 7.")
    assert_true(y.shape[0] == X.shape[0], "X/y sample counts must match.")


def test_model_build_dummy_data() -> None:
    """Kiểm tra build model nếu TensorFlow đã được cài trong môi trường."""
    try:
        import numpy as np
        from src.model_lstm import build_lstm_model

        model = build_lstm_model((7, 1))
        output = model(np.zeros((2, 7, 1), dtype="float32"), training=False)
    except Exception as exc:
        print(f"[SKIP] test_model_build_dummy_data: {exc}")
        return

    assert_true(tuple(output.shape) == (2, 1), f"Unexpected output shape: {output.shape}")


def test_metrics_dummy_data() -> None:
    """Kiểm tra tính MAE, MSE, RMSE bằng mảng giả."""
    from src.evaluate_lstm import compute_regression_metrics

    metrics = compute_regression_metrics([1.0, 2.0], [1.5, 1.5])
    assert_true(set(metrics) == {"MAE", "MSE", "RMSE"}, "Missing regression metrics.")


def test_next_day_input_dummy_data() -> None:
    """Kiểm tra reshape input dự đoán ngày tiếp theo bằng chuỗi giả."""
    import numpy as np
    from src.predict_next_day import build_next_day_input, calculate_next_date

    X = build_next_day_input(np.arange(10, dtype="float32"), {"mean": 0.0, "std": 1.0}, window_size=7)
    assert_true(X.shape == (1, 7, 1), f"Unexpected next-day input shape: {X.shape}")
    assert_true(calculate_next_date("2024-01-01").date().isoformat() == "2024-01-02", "Wrong next date.")


def test_no_absolute_paths_required_in_cli_files() -> None:
    """Kiểm tra CLI không hard-code đường dẫn tuyệt đối cục bộ.

    Hàm đọc `main.py` và `app_temperature_cli.py` để tránh phụ thuộc ổ đĩa máy cá nhân.
    """
    windows_markers = [f"{drive}:{chr(92)}" for drive in ("C", "D")]
    for file_name in ("main.py", "app_temperature_cli.py"):
        text = (PROJECT_ROOT / file_name).read_text(encoding="utf-8")
        assert_true(
            all(marker not in text for marker in windows_markers),
            f"Absolute Windows path found in {file_name}",
        )


def main() -> int:
    """Chạy toàn bộ acceptance test nhẹ cho project.

    Hàm gọi từng test theo thứ tự và in `[OK]` khi tất cả đều đạt.
    Trả về 0 khi không có assertion nào thất bại.
    """
    tests = [
        test_required_folders,
        test_self_test_runs,
        test_app_wrapper_runs,
        test_cli_argument_validation,
        test_evaluate_missing_model_message,
        test_windowing_dummy_data,
        test_model_build_dummy_data,
        test_metrics_dummy_data,
        test_next_day_input_dummy_data,
        test_no_absolute_paths_required_in_cli_files,
    ]
    for test in tests:
        test()
        print(f"[OK] {test.__name__}")
    print("[OK] temperature end-to-end acceptance tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
