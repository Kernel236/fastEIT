#!/usr/bin/env python3
"""Build supervised dataset from paired .eit/.bin Dräger recordings.

Scans a data directory of patient folders, extracts metadata via regex,
validates frame alignment, parses all pairs with fastEIT, and saves
per-patient .npz files ready for Ridge / ML training.

Usage::

    # Step 1 - filesystem scan -> recording_metadata.csv
    python -m fasteit.reconstruction.build_dataset scan --root /path/to/eit

    # Step 2 - open files, check frame counts, detect format
    python -m fasteit.reconstruction.build_dataset validate --root /path/to/eit

    # Step 3 - full parse, save per-patient .npz
    python -m fasteit.reconstruction.build_dataset build --root /path/to/eit

    # Step 4 - create sampled subset (default 7000 frames/patient)
    python -m fasteit.reconstruction.build_dataset sample --root /path/to/eit --max-frames 7000

    # All steps in sequence
    python -m fasteit.reconstruction.build_dataset all --root /path/to/eit --max-frames 7000

    # One folder per patient (ID<n>_..., each .eit with the .bin of the same name):
    # sampled .npz per patient (patient_id = 100 + n) and recordings.csv in --out
    python -m fasteit.reconstruction.build_dataset ids --root /path/to/study --out /path/to/out --max-frames 5000

Layout under ``--root``::

    draeger/PT*/                   patient folders with the .eit/.bin pairs (input)
    corrupted/                     files known to be bad, matched by name (input)
    recording_metadata.csv         one row per recording, with its status
    npz/full/patient_NN.npz        every frame of the built recordings
    npz/sample_Nk/patient_NN.npz   at most --max-frames frames per patient

Each .npz contains:
    - X_vv       : float32 (N, 208)  - calibrated transimpedances
    - X_raw      : float32 (N, 416)  - [trans_A | trans_B] concatenated
    - Y          : float32 (N, 1024) - flattened 32x32 pixel images
    - rec_id     : int16   (N,)      - recording index within patient
    - patient_id : int16   scalar
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from fasteit.reconstruction.data_prep import load_paired

METADATA_CSV = "recording_metadata.csv"
FS_HZ = 50.0  # PulmoVista 500 frame rate
BIN_FRAME_SIZES = {"EXT": 4382, "BASE": 4358}  # bytes per .bin frame, by format
EIT_FRAME_SIZE = 5495  # bytes per .eit frame
FRAME_KEYS = ("X_vv", "X_raw", "Y", "rec_id")  # the per-frame arrays of each .npz
ID_FOLDER_RE = re.compile(r"^ID(?P<n>\d+)_", re.IGNORECASE)  # one folder per patient
ID_OFFSET = 100  # patient_id = ID_OFFSET + n, apart from the PT ids (1-76)

# PT<patient>_ID<study patient>_<study>[_MALE|_FEMALE][_CONTROLLATA|_SPONT]
FOLDER_RE = re.compile(
    r"PT(?P<pt>\d+)_ID_?(?P<sid>\d+)_(?P<study>.+?)"
    r"(?:_(?P<sex>MALE|FEMALE))?(?:_(?P<mode>CONTROLLATA|SPONT))?"
)
FILE_RE = re.compile(r"pati(?:e|te)nt(?P<pt>\d+)_(?P<rec>rec\d+)")

CSV_FIELDS = [
    "patient_id",
    "study",
    "study_patient_id",
    "sex",
    "ventilation_mode",
    "recording",
    "eit_path",
    "bin_path",
    "eit_size_bytes",
    "bin_size_bytes",
    "n_frames_eit",
    "n_frames_bin",
    "bin_format",
    "fs_hz",
    "duration_s",
    "is_corrupted",
    "status",
    "notes",
]


# ===================================================================
# Step 1: SCAN
# ===================================================================


def parse_folder_name(name: str) -> dict | None:
    """Extract patient / study metadata from folder name.

    Handles patterns like:
        PT1_ID_2_RIBS_FEMALE_CONTROLLATA
        PT15_ID2_IRESP_MALE_SPONT
        PT36_ID1_ISUPPORT_SPONT
        PT56_ID1_HIFIVE
        PT57_ID1_REEF_ROMA
    """
    m = FOLDER_RE.fullmatch(name)
    if not m:
        return None
    return {
        "patient_id": int(m["pt"]),
        "study": m["study"],
        "study_patient_id": m["sid"],
        "sex": m["sex"] or "",
        "ventilation_mode": m["mode"] or "",
    }


def _corrupted_names(root: Path) -> set[str]:
    """Return set of filenames known to be corrupted."""
    corrupted_dir = root / "corrupted"
    if not corrupted_dir.exists():
        return set()
    return {f.name for f in corrupted_dir.iterdir() if f.is_file()}


def scan(root: Path) -> list[dict]:
    """Scan root/draeger and emit one row per paired recording."""
    corrupted = _corrupted_names(root)
    records: list[dict] = []

    for folder in sorted((root / "draeger").iterdir()):
        if not folder.is_dir() or not folder.name.startswith("PT"):
            continue

        meta = parse_folder_name(folder.name)
        if meta is None:
            print(f"  WARN: cannot parse folder {folder.name}, skipping")
            continue

        for eit_file in sorted(folder.glob("*.eit")):
            m = FILE_RE.match(eit_file.stem)
            if not m:
                print(f"  WARN: cannot parse filename {eit_file.name}")
                continue

            rec = m["rec"]
            bin_files = sorted(folder.glob(f"*{rec}_01.bin"))
            bin_file = bin_files[0] if bin_files else None

            is_corrupted = eit_file.name in corrupted
            if bin_file and bin_file.name in corrupted:
                is_corrupted = True

            records.append(
                {
                    **meta,
                    "recording": rec,
                    "eit_path": str(eit_file),
                    "bin_path": str(bin_file) if bin_file else "",
                    "eit_size_bytes": eit_file.stat().st_size,
                    "bin_size_bytes": bin_file.stat().st_size if bin_file else 0,
                    "is_corrupted": is_corrupted,
                    "status": "scanned",
                    "notes": "typo_filename" if "patitent" in eit_file.name else "",
                }
            )

    return records


def write_csv(records: list[dict], path: Path) -> None:
    """Write records to CSV (fields a record does not have yet are left empty)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def read_csv(path: Path) -> list[dict]:
    """Read metadata CSV back into list of dicts."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def cmd_scan(root: Path) -> None:
    """CLI: scan filesystem -> metadata CSV."""
    csv_path = root / METADATA_CSV
    print(f"Scanning {root / 'draeger'} ...")
    records = scan(root)
    write_csv(records, csv_path)

    n_corrupted = sum(1 for r in records if r["is_corrupted"])
    n_missing_bin = sum(1 for r in records if not r["bin_path"])
    studies = sorted({r["study"] for r in records})

    print(f"  {len(records)} recordings found")
    print(f"  {len({r['patient_id'] for r in records})} patients")
    print(f"  {len(studies)} studies: {', '.join(studies)}")
    print(f"  {n_corrupted} corrupted, {n_missing_bin} missing .bin")
    print(f"  Saved → {csv_path}")


# ===================================================================
# Step 2: VALIDATE
# ===================================================================


def _append_note(notes: str, note: str) -> str:
    return f"{notes}; {note}" if notes else note


def validate_record(rec: dict) -> dict:
    """Validate a single record: compute frame counts, check alignment."""
    rec = dict(rec)  # copy

    if rec["is_corrupted"] == "True":
        rec["status"] = "corrupted"
        return rec

    if not rec["bin_path"] or not Path(rec["bin_path"]).exists():
        rec["status"] = "missing_bin"
        return rec

    eit_path = Path(rec["eit_path"])
    if not eit_path.exists():
        rec["status"] = "missing_eit"
        return rec

    # .bin frame count from the file size: the first format whose frame size divides it
    bin_size = int(rec["bin_size_bytes"])
    bin_format = ""
    n_frames_bin = 0
    for fmt_name, frame_size in BIN_FRAME_SIZES.items():
        if bin_size % frame_size == 0:
            bin_format = fmt_name
            n_frames_bin = bin_size // frame_size
            break

    if not bin_format:
        rec["status"] = "invalid_bin_size"
        rec["notes"] = _append_note(rec["notes"], f"bin_size={bin_size} not divisible")
        return rec

    # .eit frame count - read preamble to find binary_start
    try:
        eit_size = int(rec["eit_size_bytes"])
        with open(eit_path, "rb") as f:
            preamble = np.frombuffer(f.read(12), dtype="<i4")
        sep_offset = int(preamble[1])
        binary_start = sep_offset + 8  # len(SEPARATOR)
        n_frames_eit = (eit_size - binary_start) // EIT_FRAME_SIZE
    except Exception as e:
        rec["status"] = "eit_header_error"
        rec["notes"] = _append_note(rec["notes"], str(e))
        return rec

    rec["n_frames_bin"] = n_frames_bin
    rec["n_frames_eit"] = n_frames_eit
    rec["bin_format"] = bin_format

    diff = abs(n_frames_eit - n_frames_bin)
    if diff > 1:
        rec["status"] = "frame_mismatch"
        rec["notes"] = _append_note(
            rec["notes"], f"eit={n_frames_eit} bin={n_frames_bin} diff={diff}"
        )
    else:
        rec["fs_hz"] = FS_HZ
        rec["duration_s"] = f"{min(n_frames_eit, n_frames_bin) / FS_HZ:.1f}"
        rec["status"] = "validated"

    return rec


def cmd_validate(root: Path) -> None:
    """CLI: validate frame counts and alignment."""
    csv_path = root / METADATA_CSV
    if not csv_path.exists():
        sys.exit("ERROR: run 'scan' first")

    records = read_csv(csv_path)
    print(f"Validating {len(records)} recordings ...")

    validated = []
    for i, rec in enumerate(records, start=1):
        validated.append(validate_record(rec))
        if i % 50 == 0:
            print(f"  {i}/{len(records)} done")

    write_csv(validated, csv_path)

    print("  Status summary:")
    for status, count in sorted(Counter(r["status"] for r in validated).items()):
        print(f"    {status}: {count}")
    print(f"  Updated → {csv_path}")


# ===================================================================
# Step 3: BUILD
# ===================================================================


def build_patient_npz(patient_id: int, records: list[dict], output_dir: Path) -> None:
    """Parse all recordings of one patient and save them in one .npz.

    Sets the status of each record to ``built`` or ``parse_error: ...``.
    """
    parts: dict[str, list[np.ndarray]] = {key: [] for key in FRAME_KEYS}

    for rec_idx, rec in enumerate(records):
        try:
            x_vv, y = load_paired(rec["eit_path"], rec["bin_path"], input_mode="vv")
            x_raw, _ = load_paired(rec["eit_path"], rec["bin_path"], input_mode="raw")
            parts["X_vv"].append(x_vv.astype(np.float32))
            parts["X_raw"].append(x_raw.astype(np.float32))
            parts["Y"].append(y.astype(np.float32))
            parts["rec_id"].append(np.full(x_vv.shape[0], rec_idx, dtype=np.int16))
            rec["status"] = "built"
        except Exception as e:
            print(f"    ERROR {rec['recording']}: {e}")
            rec["status"] = f"parse_error: {e}"

    if not parts["Y"]:
        return

    arrays = {key: np.concatenate(chunks) for key, chunks in parts.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"patient_{patient_id:02d}.npz"
    np.savez(out_path, **arrays, patient_id=np.int16(patient_id))

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(
        f"  PT{patient_id:02d}: {len(arrays['Y'])} frames, "
        f"{len(parts['Y'])} recs → {out_path.name} ({size_mb:.0f} MB)"
    )


def cmd_build(root: Path) -> None:
    """CLI: parse all validated pairs, save per-patient .npz."""
    csv_path = root / METADATA_CSV
    if not csv_path.exists():
        sys.exit("ERROR: run 'scan' and 'validate' first")

    records = read_csv(csv_path)
    buildable = [r for r in records if r["status"] == "validated"]
    print(f"Building NPZ for {len(buildable)} recordings ...")

    by_patient: dict[int, list[dict]] = {}
    for r in buildable:
        by_patient.setdefault(int(r["patient_id"]), []).append(r)

    full_dir = root / "npz" / "full"
    t0 = time.time()
    # The records are the rows of the CSV: their status is updated in place.
    for pid, patient_records in sorted(by_patient.items()):
        patient_records.sort(key=lambda r: r["recording"])
        build_patient_npz(pid, patient_records, full_dir)
    elapsed = time.time() - t0
    write_csv(records, csv_path)

    n_built = sum(1 for r in records if r["status"] == "built")
    print(f"\n  {n_built} recordings built in {elapsed:.0f}s")
    print(f"  NPZ → {full_dir}")
    print(f"  CSV → {csv_path}")


# ===================================================================
# Step 4: SAMPLE
# ===================================================================


def sample_indices(rec_id: np.ndarray, max_frames: int) -> np.ndarray:
    """Frames of a patient's sample: random within each recording, in proportion to its length.

    Seed 42 for every recording; all frames if the patient has at most ``max_frames``.
    """
    n_total = len(rec_id)
    if n_total <= max_frames:
        return np.arange(n_total)
    parts = []
    for rid in np.unique(rec_id):
        rec_frames = np.flatnonzero(rec_id == rid)
        n_sample = max(1, int(max_frames * len(rec_frames) / n_total))
        rng = np.random.default_rng(42)
        parts.append(rng.choice(rec_frames, size=n_sample, replace=False))
    # max(1, ...) can overshoot max_frames by a few frames: cut them
    return np.sort(np.concatenate(parts))[:max_frames]


def build_from_id_folders(
    source: Path, out_dir: Path, max_frames: int = 5000
) -> pd.DataFrame:
    """Sampled ``.npz`` per patient from a study with one folder per patient.

    A folder ``ID<n>_...`` is patient ``ID_OFFSET + n``; its recordings are the ``.eit``
    files (in name order) with the ``.bin`` of the same name. Other folders are
    ignored. Writes ``patient_<id>.npz`` (``FRAME_KEYS`` and ``patient_id``) and
    ``recordings.csv`` (``patient_id``, ``id``, ``recording``, ``n_frames``, ``status`` =
    ``used``, ``no_bin`` or ``frame_mismatch``), with no file names.

    Returns:
        The table written as ``recordings.csv``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for folder in sorted(p for p in source.iterdir() if p.is_dir()):
        match = ID_FOLDER_RE.match(folder.name)
        if not match:
            continue
        n = int(match["n"])
        patient_id = ID_OFFSET + n
        parts: dict[str, list[np.ndarray]] = {key: [] for key in FRAME_KEYS}
        for k, eit_path in enumerate(sorted(folder.glob("*.eit"))):
            row = {
                "patient_id": patient_id,
                "id": f"ID{n}",
                "recording": k,
                "n_frames": 0,
            }
            rows.append(row)
            bin_path = eit_path.with_suffix(".bin")
            if not bin_path.exists():
                row["status"] = "no_bin"
                continue
            try:
                x_vv, y = load_paired(eit_path, bin_path, input_mode="vv")
                x_raw, _ = load_paired(eit_path, bin_path, input_mode="raw")
            except ValueError:
                row["status"] = "frame_mismatch"
                continue
            row |= {"n_frames": len(y), "status": "used"}
            for key, values in (("X_vv", x_vv), ("X_raw", x_raw), ("Y", y)):
                parts[key].append(np.asarray(values, dtype=np.float32))
            parts["rec_id"].append(np.full(len(y), k, dtype=np.int16))
        if parts["Y"]:
            arrays = {key: np.concatenate(values) for key, values in parts.items()}
            idx = sample_indices(arrays["rec_id"], max_frames)
            np.savez(
                out_dir / f"patient_{patient_id}.npz",
                **{key: arrays[key][idx] for key in FRAME_KEYS},
                patient_id=np.int16(patient_id),
            )
    columns = ["patient_id", "id", "recording", "n_frames", "status"]
    table = pd.DataFrame(rows, columns=columns)
    table.to_csv(out_dir / "recordings.csv", index=False)
    return table


def cmd_sample(root: Path, max_frames: int) -> None:
    """CLI: create sampled subset from full .npz files.

    Frames are drawn at random (sorted, seed 42) within each recording, in proportion to
    its length, so the frames of a sample are not contiguous in time.
    """
    full_dir = root / "npz" / "full"
    npz_files = sorted(full_dir.glob("patient_*.npz"))
    if not npz_files:
        sys.exit("ERROR: run 'build' first - no .npz files in full/")

    sample_dir = root / "npz" / f"sample_{max_frames // 1000}k"
    sample_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sampling max {max_frames} frames/patient from {len(npz_files)} files ...")
    total_frames = 0

    for npz_path in npz_files:
        data = np.load(npz_path)
        rec_id = data["rec_id"]
        n_total = len(rec_id)

        idx = sample_indices(rec_id, max_frames)

        np.savez(
            sample_dir / npz_path.name,
            **{key: data[key][idx] for key in FRAME_KEYS},
            patient_id=data["patient_id"],
        )
        total_frames += len(idx)
        print(f"  {npz_path.stem}: {n_total} → {len(idx)} frames")

    size_gb = sum(f.stat().st_size for f in sample_dir.glob("*.npz")) / 1e9
    print(f"\n  Total: {total_frames} frames ({size_gb:.1f} GB)")
    print(f"  Saved → {sample_dir}")


# ===================================================================
# CLI
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build supervised EIT dataset from paired .eit/.bin files."
    )
    parser.add_argument(
        "step",
        choices=["scan", "validate", "build", "sample", "all", "ids"],
        help="Pipeline step to run.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Data root with draeger/, corrupted/, npz/ and recording_metadata.csv "
        "(ids: the study folder with one ID<n>_... folder per patient).",
    )
    parser.add_argument("--out", type=Path, help="Output folder of the ids step.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=7_000,
        help="Max frames per patient in sample step (default: 7000).",
    )

    args = parser.parse_args()

    if args.step in ("scan", "all"):
        cmd_scan(args.root)
    if args.step in ("validate", "all"):
        cmd_validate(args.root)
    if args.step in ("build", "all"):
        cmd_build(args.root)
    if args.step in ("sample", "all"):
        cmd_sample(args.root, args.max_frames)
    if args.step == "ids":
        if args.out is None:
            sys.exit("ERROR: the ids step needs --out")
        table = build_from_id_folders(args.root, args.out, args.max_frames)
        print(table.groupby("status").size().to_string())


if __name__ == "__main__":
    main()
