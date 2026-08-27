"""GPU detection — CUDA, Metal, ROCm support checking."""

import platform
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUInfo:
    name: str
    vram_mb: int
    backend: str  # cuda, metal, rocm, cpu
    compute_capability: Optional[str] = None


def check_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def check_metal() -> bool:
    if platform.system() != "Darwin":
        return False
    try:
        import torch
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        return False


def check_rocm() -> bool:
    try:
        import torch
        return hasattr(torch.version, "hip") and torch.version.hip is not None
    except ImportError:
        return False


def get_gpu_info() -> Optional[GPUInfo]:
    if check_cuda():
        try:
            import torch
            idx = torch.cuda.current_device()
            return GPUInfo(
                name=torch.cuda.get_device_name(idx),
                vram_mb=torch.cuda.get_device_properties(idx).total_mem // (1024 * 1024),
                backend="cuda",
                compute_capability=f"{torch.cuda.get_device_capability(idx)[0]}.{torch.cuda.get_device_capability(idx)[1]}",
            )
        except Exception:
            pass
    if check_metal():
        return GPUInfo(name="Apple Silicon (MPS)", vram_mb=0, backend="metal")
    if check_rocm():
        return GPUInfo(name="AMD GPU (ROCm)", vram_mb=0, backend="rocm")
    return None


def get_recommended_backend() -> str:
    gpu = get_gpu_info()
    if gpu is None:
        return "llama_cpp"  # CPU inference
    if gpu.backend == "cuda":
        try:
            from llama_cpp import Llama
            return "llama_cpp"
        except ImportError:
            return "transformers"
    return "llama_cpp"


def format_gpu_info() -> str:
    gpu = get_gpu_info()
    if gpu is None:
        return "No GPU detected — using CPU inference"
    vram = f"{gpu.vram_mb} MB" if gpu.vram_mb > 0 else "Shared memory"
    cc = f" (compute {gpu.compute_capability})" if gpu.compute_capability else ""
    return f"{gpu.name} — {vram} [{gpu.backend}]{cc}"
