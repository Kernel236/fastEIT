#!/usr/bin/env python3
"""Build supervised dataset from paired .eit/.bin Dräger recordings.

Scans a data directory of patient folders, extracts metadata via regex,
validates frame alignment, parses all pairs with fastEIT, and saves
per-patient .npz files ready for Ridge / ML training.

Usage::

    # Step 1 — filesystem scan → recording_metadata.csv
    python -m fasteit.reconstruction.build_dataset scan

    # Step 2 — open files, check frame counts, detect format
    python -m fasteit.reconstruction.build_dataset validate

    # Step 3 — full parse, save per-patient .npz
    python -m fasteit.reconstruction.build_dataset build

    # Step 4 — create sampled subset (default 7 000 frames/patient)
    python -m fasteit.reconstruction.build_dataset sample --max-frames 7000

    # All steps in sequence
    python -m fasteit.reconstruction.build_dataset all --max-frames 7000

Output structure::

    {OUTPUT_ROOT}/
    ├── recording_metadata.csv
    ├── full/
    │   ├── patient_01.npz
    │   └── ...
    └── sample_10k/
        ├── patient_01.npz
        └── ...

Each .npz contains:
    - X_vv    : float32 (N, 208)  — calibrated transimpedances
    - X_raw   : float32 (N, 416)  — [trans_A | trans_B] concatenated
    - Y       : float32 (N, 1024) — flattened 32×32 pixel images
    - rec_id  : int16   (N,)      — recording index within patient
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration — edit these paths for your machine
# ---------------------------------------------------------------------------

DATA_ROOT = Path("/mnt/my_data/eit/draeger")
CORRUPTED_DIR = Path("/mnt/my_data/eit/corrupted")
OUTPUT_ROOT = Path("/mnt/my_data/eit/npz")
METADATA_CSV = Path("/mnt/my_data/eit/recording_metadata.csv")

BIN_FRAME_SIZES = {"EXT": 4382, "BASE": 4358}
EIT_FRAME_SIZE = 5495

# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------

FOLDER_RE = re.compile(r"PT(?P<pt>\d+)_ID_?(?P<sid>\d+)_(?P<rest>.+)")
FILE_RE = re.compile(r"pati(?:e|te)nt(?P<pt>\d+)_(?P<rec>rec\d+)")

# ---------------------------------------------------------------------------
# CSV field order
# ---------------------------------------------------------------------------

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
    m = FOLDER_RE.match(name)
    if not m:
        return None

    pt = int(m.group("pt"))
    sid = m.group("sid")
    rest = m.group("rest")

    sex = None
    mode = None

    if rest.endswith("_CONTROLLATA"):
        mode = "CONTROLLATA"
        rest = rest[: -len("_CONTROLLATA")]
    elif rest.endswith("_SPONT"):
        mode = "SPONT"
        rest = rest[: -len("_SPONT")]

    if rest.endswith("_MALE"):
        sex = "MALE"
        rest = rest[: -len("_MALE")]
    elif rest.endswith("_FEMALE"):
        sex = "FEMALE"
        rest = rest[: -len("_FEMALE")]

    return {
        "patient_id": pt,
        "study": rest,
        "study_patient_id": sid,
        "sex": sex or "",
        "ventilation_mode": mode or "",
    }


def _corrupted_names() -> set[str]:
    """Return set of filenames known to be corrupted."""
    if not CORRUPTED_DIR.exists():
        return set()
    return {f.name for f in CORRUPTED_DIR.iterdir() if f.is_file()}


def scan() -> list[dict]:
    """Scan DATA_ROOT and emit one row per paired recording."""
    corrupted = _corrupted_names()
    records: list[dict] = []

    for folder in sorted(DATA_ROOT.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("PT"):
            continue

        meta = parse_folder_name(folder.name)
        if meta is None:
            print(f"  WARN: cannot parse folder {folder.name}, skipping")
            continue

        eit_files = sorted(folder.glob("*.eit"))

        for eit_file in eit_files:
            m = FILE_RE.match(eit_file.stem)
            if not m:
                print(f"  WARN: cannot parse filename {eit_file.name}")
                continue

            rec = m.group("rec")

            # Find matching .bin — try *rec01_01.bin first
            bin_cands = sorted(folder.glob(f"*{rec}_01.bin"))
            bin_file = bin_cands[0] if bin_cands else None

            notes: list[str] = []
            if "patitent" in eit_file.name:
                notes.append("typo_filename")

            is_corrupted = eit_file.name in corrupted
            if bin_file and bin_file.name in corrupted:
                is_corrupted = True

            record = {
                **meta,
                "recording": rec,
                "eit_path": str(eit_file),
                "bin_path": str(bin_file) if bin_file else "",
                "eit_size_bytes": eit_file.stat().st_size,
                "bin_size_bytes": bin_file.stat().st_size if bin_file else 0,
                "n_frames_eit": "",
                "n_frames_bin": "",
                "bin_format": "",
                "fs_hz": "",
                "duration_s": "",
                "is_corrupted": is_corrupted,
                "status": "scanned",
                "notes": "; ".join(notes),
            }
            records.append(record)

    return records


def write_csv(records: list[dict], path: Path) -> None:
    """Write records to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def read_csv(path: Path) -> list[dict]:
    """Read metadata CSV back into list of dicts."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def cmd_scan() -> None:
    """CLI: scan filesystem → metadata CSV."""
    print(f"Scanning {DATA_ROOT} ...")
    records = scan()
    write_csv(records, METADATA_CSV)

    n_corrupted = sum(1 for r in records if r["is_corrupted"])
    n_missing_bin = sum(1 for r in records if not r["bin_path"])
    studies = sorted({r["study"] for r in records})

    print(f"  {len(records)} recordings found")
    print(f"  {len({r['patient_id'] for r in records})} patients")
    print(f"  {len(studies)} studies: {', '.join(studies)}")
    print(f"  {n_corrupted} corrupted, {n_missing_bin} missing .bin")
    print(f"  Saved → {METADATA_CSV}")


# ===================================================================
# Step 2: VALIDATE
# ===================================================================


def validate_record(rec: dict) -> dict:
    """Validate a single record: compute frame counts, check alignment."""
    rec = dict(rec)  # copy

    if rec["is_corrupted"] in (True, "True", "true"):
        rec["status"] = "corrupted"
        return rec

    eit_path = Path(rec["eit_path"])
    bin_path = Path(rec["bin_path"]) if rec["bin_path"] else None

    if not bin_path or not bin_path.exists():
        rec["status"] = "missing_bin"
        return rec

    if not eit_path.exists():
        rec["status"] = "missing_eit"
        return rec

    # .bin frame count from file size
    bin_size = int(rec["bin_size_bytes"])
    bin_format = ""
    n_frames_bin = 0
    for fmt_name, fs in BIN_FRAME_SIZES.items():
        if bin_size % fs == 0:
            bin_format = fmt_name
            n_frames_bin = bin_size // fs
            break

    if not bin_format:
        rec["status"] = "invalid_bin_size"
        rec["notes"] = _append_note(rec["notes"], f"bin_size={bin_size} not divisible")
        return rec

    # .eit frame count — read preamble to find binary_start
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

    # Frame count alignment check
    diff = abs(n_frames_eit - n_frames_bin)
    if diff > 1:
        rec["status"] = "frame_mismatch"
        rec["notes"] = _append_note(
            rec["notes"], f"eit={n_frames_eit} bin={n_frames_bin} diff={diff}"
        )
    else:
        rec["fs_hz"] = "50.0"
        n_min = min(n_frames_eit, n_frames_bin)
        rec["duration_s"] = f"{n_min / 50.0:.1f}"
        rec["status"] = "validated"

    return rec


def _append_note(existing: str, new: str) -> str:
    if existing:
        return f"{existing}; {new}"
    return new


def cmd_validate() -> None:
    """CLI: validate frame counts and alignment."""
    if not METADATA_CSV.exists():
        print("ERROR: run 'scan' first")
        sys.exit(1)

    records = read_csv(METADATA_CSV)
    print(f"Validating {len(records)} recordings ...")

    validated = []
    for i, rec in enumerate(records):
        rec = validate_record(rec)
        validated.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(records)} done")

    write_csv(validated, METADATA_CSV)

    # Summary
    by_status: dict[str, int] = {}
    for r in validated:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print("  Status summary:")
    for status, count in sorted(by_status.items()):
        print(f"    {status}: {count}")
    print(f"  Updated → {METADATA_CSV}")


# ===================================================================
# Step 3: BUILD
# ===================================================================


def build_patient_npz(
    patient_id: int,
    patient_records: list[dict],
    output_dir: Path,
) -> dict[str, str]:
    """Parse all recordings for one patient, save .npz.

    Returns dict of {recording: status} for CSV update.
    """
    from fasteit.reconstruction.data_prep import load_paired

    all_x_vv: list[np.ndarray] = []
    all_x_raw: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_rec_id: list[np.ndarray] = []
    rec_statuses: dict[str, str] = {}

    for rec_idx, rec in enumerate(patient_records):
        eit_path = Path(rec["eit_path"])
        bin_path = Path(rec["bin_path"])
        rec_label = rec["recording"]

        try:
            x_vv, y = load_paired(eit_path, bin_path, input_mode="vv")
            x_raw, _ = load_paired(eit_path, bin_path, input_mode="raw")

            all_x_vv.append(x_vv.astype(np.float32))
            all_x_raw.append(x_raw.astype(np.float32))
            all_y.append(y.astype(np.float32))
            all_rec_id.append(np.full(x_vv.shape[0], rec_idx, dtype=np.int16))
            rec_statuses[rec_label] = "built"

        except Exception as e:
            print(f"    ERROR {rec_label}: {e}")
            rec_statuses[rec_label] = f"parse_error: {e}"

    if not all_x_vv:
        return rec_statuses

    X_vv = np.concatenate(all_x_vv)
    X_raw = np.concatenate(all_x_raw)
    Y = np.concatenate(all_y)
    rec_id = np.concatenate(all_rec_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"patient_{patient_id:02d}.npz"

    np.savez(
        out_path,
        X_vv=X_vv,
        X_raw=X_raw,
        Y=Y,
        rec_id=rec_id,
        patient_id=np.int16(patient_id),
    )

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(
        f"  PT{patient_id:02d}: {X_vv.shape[0]} frames, "
        f"{len(all_x_vv)} recs → {out_path.name} ({size_mb:.0f} MB)"
    )

    return rec_statuses


def cmd_build() -> None:
    """CLI: parse all pairs, save per-patient .npz."""
    if not METADATA_CSV.exists():
        print("ERROR: run 'scan' and 'validate' first")
        sys.exit(1)

    records = read_csv(METADATA_CSV)

    # Filter to validated, non-corrupted recordings with both files
    buildable = [
        r
        for r in records
        if r["status"] == "validated" and r["bin_path"]
    ]
    print(f"Building NPZ for {len(buildable)} recordings ...")

    # Group by patient_id
    by_patient: dict[int, list[dict]] = {}
    for r in buildable:
        pid = int(r["patient_id"])
        by_patient.setdefault(pid, []).append(r)

    full_dir = OUTPUT_ROOT / "full"
    t0 = time.time()

    for i, (pid, precs) in enumerate(sorted(by_patient.items())):
        precs_sorted = sorted(precs, key=lambda r: r["recording"])
        statuses = build_patient_npz(pid, precs_sorted, full_dir)

        # Update CSV records
        for r in records:
            if int(r["patient_id"]) == pid and r["recording"] in statuses:
                s = statuses[r["recording"]]
                r["status"] = s if s != "built" else "built"

    elapsed = time.time() - t0
    write_csv(records, METADATA_CSV)

    n_built = sum(1 for r in records if r["status"] == "built")
    print(f"\n  {n_built} recordings built in {elapsed:.0f}s")
    print(f"  NPZ → {full_dir}")
    print(f"  CSV → {METADATA_CSV}")


# ===================================================================
# Step 4: SAMPLE
# ===================================================================


def cmd_sample(max_frames: int = 10_000) -> None:
    """CLI: create sampled subset from full .npz files."""
    full_dir = OUTPUT_ROOT / "full"
    sample_dir = OUTPUT_ROOT / f"sample_{max_frames // 1000}k"
    sample_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(full_dir.glob("patient_*.npz"))
    if not npz_files:
        print("ERROR: run 'build' first — no .npz files in full/")
        sys.exit(1)

    print(f"Sampling max {max_frames} frames/patient from {len(npz_files)} files ...")
    total_frames = 0

    for npz_path in npz_files:
        data = np.load(npz_path)
        n_total = data["X_vv"].shape[0]

        if n_total <= max_frames:
            # Keep all frames
            idx = np.arange(n_total)
        else:
            # Stratified sampling: proportional per recording
            rec_id = data["rec_id"]
            unique_recs = np.unique(rec_id)
            idx_parts: list[np.ndarray] = []

            for rid in unique_recs:
                rec_mask = rec_id == rid
                rec_indices = np.where(rec_mask)[0]
                # Proportional allocation
                n_sample = max(
                    1, int(max_frames * len(rec_indices) / n_total)
                )
                n_sample = min(n_sample, len(rec_indices))
                chosen = np.sort(
                    np.random.default_rng(42).choice(
                        rec_indices, size=n_sample, replace=False
                    )
                )
                idx_parts.append(chosen)

            idx = np.sort(np.concatenate(idx_parts))
            # Trim to exact max_frames if rounding caused overshoot
            if len(idx) > max_frames:
                idx = idx[:max_frames]

        out_path = sample_dir / npz_path.name
        np.savez(
            out_path,
            X_vv=data["X_vv"][idx],
            X_raw=data["X_raw"][idx],
            Y=data["Y"][idx],
            rec_id=data["rec_id"][idx],
            patient_id=data["patient_id"],
        )

        total_frames += len(idx)
        print(
            f"  {npz_path.stem}: {n_total} → {len(idx)} frames"
        )

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
        choices=["scan", "validate", "build", "sample", "all"],
        help="Pipeline step to run.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=7_000,
        help="Max frames per patient in sample step (default: 7000).",
    )

    args = parser.parse_args()

    if args.step == "scan":
        cmd_scan()
    elif args.step == "validate":
        cmd_validate()
    elif args.step == "build":
        cmd_build()
    elif args.step == "sample":
        cmd_sample(args.max_frames)
    elif args.step == "all":
        cmd_scan()
        cmd_validate()
        cmd_build()
        cmd_sample(args.max_frames)


if __name__ == "__main__":
    main()
