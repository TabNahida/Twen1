from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load_script("check_cuda_toolchain.py")


def _fake_toolkit(root: Path, *, nvcc: str = "13.2", cudart: int = 13020) -> Path:
    (root / "bin").mkdir(parents=True)
    (root / "include" / "cccl" / "cuda" / "std" / "__cccl").mkdir(parents=True)
    nvcc_path = root / "bin" / "nvcc"
    nvcc_path.write_text(
        f"#!/usr/bin/env bash\necho 'Cuda compilation tools, release {nvcc}, V{nvcc}.51'\n",
        encoding="utf-8",
    )
    nvcc_path.chmod(0o700)
    (root / "include" / "cuda_runtime_api.h").write_text(
        f"#define CUDART_VERSION {cudart}\n",
        encoding="utf-8",
    )
    (root / "include" / "cccl" / "cuda" / "std" / "__cccl" / "version.h").write_text(
        "#define CCCL_VERSION 3002000\n",
        encoding="utf-8",
    )
    return root


def test_toolchain_inspection_accepts_matching_compiler_and_runtime_headers(
    tmp_path: Path,
) -> None:
    report = checker.inspect_toolchain(_fake_toolkit(tmp_path / "cuda"))

    assert report["nvcc_release"] == "13.2"
    assert report["cudart_release"] == "13.2"
    assert report["cccl_release"] == "3.2.0"
    assert report["compiler_headers_match"] is True


def test_toolchain_inspection_exposes_mixed_cuda_wheel_stack(tmp_path: Path) -> None:
    report = checker.inspect_toolchain(_fake_toolkit(tmp_path / "cuda", nvcc="13.2", cudart=13000))

    assert report["nvcc_release"] == "13.2"
    assert report["cudart_release"] == "13.0"
    assert report["compiler_headers_match"] is False


def test_fla_smoke_and_launcher_never_create_an_optimizer() -> None:
    root = Path(__file__).parents[1]
    smoke = (root / "scripts" / "smoke_fla_backward.py").read_text(encoding="utf-8")
    launcher = (root / "scripts" / "with_cuda_toolchain.sh").read_text(encoding="utf-8")

    assert "torch.optim" not in smoke
    assert "optimizer.step(" not in smoke
    assert "LD_LIBRARY_PATH" not in launcher
    assert 'export CUDA_HOME="${cuda_home}"' in launcher
    assert 'export FLA_TILELANG="${FLA_TILELANG:-0}"' in launcher
