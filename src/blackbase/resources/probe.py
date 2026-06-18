"""
Local L0 worker/resource discovery helpers.

Provides utilities for detecting local compute resources and building
worker descriptors for single-machine execution.
"""

from __future__ import annotations

import ctypes
import os
import socket
from typing import Any, Dict, Mapping, Optional

from .model import ResourceOffer, WorkerDescriptor


def detect_total_memory_mb() -> Optional[float]:
    """
    Best-effort physical memory detection without mandatory dependencies.
    
    Falls back gracefully if psutil or platform-specific APIs are unavailable.
    """
    # Try psutil first
    try:
        import psutil  # type: ignore
        return float(psutil.virtual_memory().total) / (1024.0 * 1024.0)
    except Exception:
        pass
    
    # Windows-specific detection
    if os.name == "nt":
        try:
            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            
            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                return float(status.ullTotalPhys) / (1024.0 * 1024.0)
        except Exception:
            return None
    
    # Unix-like systems
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return float(pages * page_size) / (1024.0 * 1024.0)
    except Exception:
        return None


def detect_cuda_devices() -> tuple[tuple[str, ...], Dict[str, float], Dict[str, Any]]:
    """
    Best-effort CUDA token and VRAM detection.
    
    Uses torch when available. Without torch, falls back to
    CUDA_VISIBLE_DEVICES as token hints but cannot infer VRAM.
    
    Returns:
        Tuple of (device_tokens, vram_by_device, metadata)
    """
    metadata: Dict[str, Any] = {}
    
    # Try PyTorch
    try:
        import torch  # type: ignore
        
        if bool(torch.cuda.is_available()):
            count = int(torch.cuda.device_count())
            tokens = tuple(f"cuda:{idx}" for idx in range(count))
            vram: Dict[str, float] = {}
            names: Dict[str, str] = {}
            for idx in range(count):
                props = torch.cuda.get_device_properties(idx)
                token = f"cuda:{idx}"
                vram[token] = float(getattr(props, "total_memory", 0.0)) / (1024.0 * 1024.0)
                names[token] = str(getattr(props, "name", token))
            metadata["cuda_source"] = "torch"
            metadata["gpu_names"] = names
            return tokens, vram, metadata
    except Exception as exc:
        metadata["cuda_probe_error"] = f"{type(exc).__name__}: {exc}"
    
    # Fallback to environment variable
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or visible in {"-1", "none", "None"}:
        return tuple(), {}, metadata
    
    raw_tokens = tuple(x.strip() for x in visible.split(",") if x.strip())
    tokens = tuple(f"cuda:{idx}" for idx, _ in enumerate(raw_tokens))
    metadata["cuda_source"] = "CUDA_VISIBLE_DEVICES"
    metadata["cuda_visible_devices"] = list(raw_tokens)
    return tokens, {}, metadata


def detect_local_resource_offer(
    *,
    threads: Optional[int] = None,
    include_cuda: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ResourceOffer:
    """
    Detect local machine resources and create a ResourceOffer.
    
    Args:
        threads: Override CPU thread count (defaults to os.cpu_count())
        include_cuda: Whether to probe for CUDA devices
        metadata: Additional metadata to include
    
    Returns:
        ResourceOffer with detected local resources
    """
    cpu_threads = int(threads or os.cpu_count() or 1)
    mem_mb = detect_total_memory_mb()
    
    meta: Dict[str, Any] = {
        "host": socket.gethostname(),
        "hardware_backend": "local_machine",
    }
    if mem_mb is not None:
        meta["memory_mb"] = float(mem_mb)
    if metadata:
        meta.update(dict(metadata))
    
    device_tokens: tuple[str, ...] = tuple()
    if include_cuda:
        tokens, vram_by_device, cuda_meta = detect_cuda_devices()
        device_tokens = tuple(tokens)
        meta.update(cuda_meta)
        if vram_by_device:
            meta["gpu_memory_mb_by_device"] = dict(vram_by_device)
            meta["gpu_memory_mb"] = float(sum(vram_by_device.values()))
    
    return ResourceOffer(
        threads=max(1, cpu_threads),
        gpus=len(device_tokens),
        backend="local",
        device_tokens=device_tokens,
        metadata=meta,
    )


def build_local_worker_descriptor(
    *,
    worker_id: str = "local-worker",
    executor_backend: str = "thread",
    capabilities: tuple[str, ...] = ("cpu", "numpy"),
    include_cuda: bool = True,
    max_inflight: int = 1,
    threads: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> WorkerDescriptor:
    """
    Build a WorkerDescriptor for local machine execution.
    
    Args:
        worker_id: Unique identifier for the worker
        executor_backend: Backend for task execution (thread/process)
        capabilities: Set of capabilities this worker supports
        include_cuda: Whether to probe for CUDA devices
        max_inflight: Maximum number of concurrent tasks
        threads: Override CPU thread count
        metadata: Additional metadata
    
    Returns:
        WorkerDescriptor for local execution
    """
    offer = detect_local_resource_offer(
        threads=threads,
        include_cuda=include_cuda,
        metadata=metadata,
    )
    
    caps = set(str(x) for x in capabilities)
    if offer.gpus > 0:
        caps.update({"gpu", "cuda"})
    
    return WorkerDescriptor(
        worker_id=str(worker_id),
        executor_backend=str(executor_backend),
        resource_backend="local",
        host=str(offer.metadata.get("host", socket.gethostname())),
        capabilities=tuple(sorted(caps)),
        offer=offer,
        max_inflight=max(1, int(max_inflight)),
        metadata={"probe": "local"},
    )