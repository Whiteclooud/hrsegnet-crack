#!/usr/bin/env python3
"""Print the runtime environment used for HrSegNet experiments."""

from __future__ import annotations

import importlib
import platform
import subprocess
import sys


def version_of(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - diagnostic script
        return f"not available ({exc})"
    return getattr(module, "__version__", "unknown")


def command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError:
        return "not available"
    return result.stdout.strip() or "no output"


def main() -> None:
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"paddle: {version_of('paddle')}")
    print(f"paddleseg: {version_of('paddleseg')}")
    print(f"cv2: {version_of('cv2')}")
    print(f"yaml: {version_of('yaml')}")
    print("nvidia-smi:")
    print(command_output(["nvidia-smi"]))


if __name__ == "__main__":
    main()
