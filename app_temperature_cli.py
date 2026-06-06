from __future__ import annotations

import sys

from main import build_parser, main


def app_main(argv: list[str] | None = None) -> int:
    """File bao lệnh đơn giản cho CLI chính.

    Tham số `argv` là danh sách lệnh như `self_test`, `preprocess`, `train`.
    Hàm trả về mã trạng thái từ `main.py` và không thay đổi logic xử lý.
    """
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        parser = build_parser()
        parser.print_help()
        return 0

    return main(argv)


if __name__ == "__main__":
    sys.exit(app_main())
