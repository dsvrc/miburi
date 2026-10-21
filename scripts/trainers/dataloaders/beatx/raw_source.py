"""Build BEATX chunks straight from the raw files, in memory, at dataset init.

Alternative to the HDF5 cache (``--beatx_source raw``): runs the same per-file
processing as ``build_hdf5_beatx.py`` but keeps the chunks in RAM, so nothing
has to be built beforehand. Builds are memoised per process, so the
train/val/test datasets of one run share a single build.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import numpy as np
import torch
from loguru import logger

from . import build_hdf5_beatx as builder
from ..utils.data_tools import check_if_additional_split


class _MemDataset:
    """h5py.Dataset stand-in: indexing returns a copy, like a read from disk."""

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def __getitem__(self, idx):
        return np.array(self._arr[idx])


class _MemGroup:
    def __init__(self, attrs: dict[str, Any], datasets: dict[str, np.ndarray]):
        self.attrs = attrs
        self._datasets = {name: _MemDataset(arr) for name, arr in datasets.items()}

    def __contains__(self, name: str) -> bool:
        return name in self._datasets

    def __getitem__(self, name: str) -> _MemDataset:
        return self._datasets[name]


class BeatxMemoryStore:
    """Chunks of one raw BEATX build, addressable like an open HDF5 file."""

    def __init__(self, key: str):
        self.key = key
        self.groups: dict[str, _MemGroup] = {}
        self.chunk_ids: list[str] = []
        self.chunk_splits: list[str] = []
        self.chunk_relpaths: list[str] = []

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self.groups

    def __getitem__(self, chunk_id: str) -> _MemGroup:
        return self.groups[chunk_id]

    def close(self):
        pass


_STORES: dict[str, BeatxMemoryStore] = {}


def get_beatx_memory_store(
    args,
    *,
    speakers: Optional[list[str]],
    full_sequence: bool,
    multimodal: bool,
    exclude_additional: bool,
) -> BeatxMemoryStore:
    data_dir = args.beatx_data_path
    split_csv = getattr(args, "beatx_split_csv", None) or os.path.join(data_dir, "train_test_split.csv")
    spk_key = ",".join(sorted(speakers)) if speakers else "all"
    key = (
        f"memory:{data_dir}|{split_csv}|{spk_key}|full={int(full_sequence)}|mm={int(multimodal)}"
        f"|noadd={int(exclude_additional)}|{args.pose_length}@{args.motion_fps}"
    )
    if key not in _STORES:
        _STORES[key] = _build_store(
            key, args, data_dir, split_csv, speakers, full_sequence, multimodal, exclude_additional
        )
    return _STORES[key]


def _build_store(
    key: str,
    args,
    data_dir: str,
    split_csv: str,
    speakers: Optional[list[str]],
    full_sequence: bool,
    multimodal: bool,
    exclude_additional: bool,
) -> BeatxMemoryStore:
    bargs = builder.build_arg_parser().parse_args([
        "--hdf5_path", key,
        "--data_dir", data_dir,
        "--file_list_path", split_csv,
        "--motion_fps", str(args.motion_fps),
        "--pose_length", str(args.pose_length),
        "--smplx_model_path", os.path.join(args.deps_path, "smplx_2020/"),
    ])
    bargs.full_sequence = full_sequence
    bargs.process_multimodal_signals = multimodal

    entries = builder._build_file_list(bargs)
    if speakers:
        wanted = set(speakers)
        entries = [e for e in entries if builder._beatx_speaker_name(e.file_id) in wanted]
    if exclude_additional:
        entries = [e for e in entries if not check_if_additional_split(e.file_id)]
    if not entries:
        raise ValueError(
            f"No BEATX files for speakers={speakers or 'all'} in {split_csv} under {data_dir} "
            "(needs smplxflame_25/, wave16k/ and whisper_transcription/ for each take)."
        )

    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    logger.info(
        "Raw BEATX build: {} files, speakers={}, full_sequence={}, multimodal={}",
        len(entries), speakers or "all", full_sequence, multimodal,
    )
    models = builder._load_models(bargs, device, load_mimi=multimodal)

    store = BeatxMemoryStore(key)
    statuses: dict[str, int] = {}
    t0 = time.time()
    with torch.no_grad():
        for i, entry in enumerate(entries):
            result = builder._process_entry(entry, bargs, models)
            status = result.get("status", "ok")
            statuses[status] = statuses.get(status, 0) + 1
            for chunk in result.get("chunks", []):
                attrs = chunk["attrs"]
                store.groups[chunk["filechunk_id"]] = _MemGroup(dict(attrs), chunk["datasets"])
                store.chunk_ids.append(chunk["filechunk_id"])
                store.chunk_splits.append(attrs["split"])
                store.chunk_relpaths.append(chunk["relpath"])
            if (i + 1) % 25 == 0 or i + 1 == len(entries):
                logger.info(
                    "Raw BEATX build: {}/{} files, {} chunks, {:.0f}s",
                    i + 1, len(entries), len(store.chunk_ids), time.time() - t0,
                )

    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Raw BEATX build done: {} chunks, file statuses {}", len(store.chunk_ids), statuses)
    return store
