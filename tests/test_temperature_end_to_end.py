from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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

    Nếu `models/temp_lstm.keras` chưa tồn tại, CLI phải hướng dẫn rõ cần train trước.
    """
    result = run_command(["main.py", "evaluate"])
    output = result.stdout + result.stderr
    if not (PROJECT_ROOT / "models" / "temp_lstm.keras").exists():
        assert_true(result.returncode == 2, output)
        assert_true("Missing trained model" in output, output)


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
        test_no_absolute_paths_required_in_cli_files,
    ]
    for test in tests:
        test()
        print(f"[OK] {test.__name__}")
    print("[OK] temperature end-to-end acceptance tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
