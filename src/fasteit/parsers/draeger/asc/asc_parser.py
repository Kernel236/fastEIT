"""Drager `.asc` parser for PulmoVista tabular exports."""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

from fasteit.models.continuous_data import ContinuousSignalData
from fasteit.parsers.base import BaseParser
from fasteit.parsers.detection import detect_vendor_from_tabular



def _to_snake_case(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", text.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def _split_tab_fields(line: str) -> list[str]:
    """Split one tab-separated line and drop trailing empty fields."""
    return [field.strip() for field in line.rstrip("\n").split("\t") if field.strip() != ""]


def _extract_header_metadata(lines: list[str]) -> dict:
    metadata: dict[str, str | int | float] = {}

    for raw in lines:
        line = raw.strip()
        lower = line.lower()

        if lower.startswith("file:"):
            metadata["source_eit_file"] = line.split(":", 1)[1].strip()
        elif lower.startswith("length:"):
            match = re.search(r"(\d+)\s+images.*=\s*(\d+)\s*s", lower)
            if match:
                metadata["declared_images"] = int(match.group(1))
                metadata["declared_duration_s"] = int(match.group(2))
        elif lower.startswith("dynamic image, time:"):
            value = line.split(":", 1)[1].strip().replace(",", ".")
            try:
                metadata["dynamic_image_time"] = float(value)
            except ValueError:
                metadata["dynamic_image_time"] = value
        elif lower.startswith("lp/bp-filter:"):
            metadata["filter_mode"] = line.split(":", 1)[1].strip()
        elif lower.startswith("filter cut-off frequ:"):
            metadata["filter_cutoff"] = line.split(":", 1)[1].strip()

    return metadata


class DragerAscParser(BaseParser):
    """Parser for Drager PulmoVista `.asc` exports.

    Metadata from the ASCII header (file name, recording length, filter settings)
    is preserved in the returned ContinuousSignalData.metadata dict.
    """

    def validate(self, path: Path) -> bool:
        path = Path(path)
        if not path.exists() or path.stat().st_size == 0:
            return False
        if path.suffix.lower() not in {".asc", ".txt", ".csv"}:
            return False

        try:
            vendor = detect_vendor_from_tabular(path)
        except ValueError:
            return False
        return vendor == "draeger"

    def parse(self, path: Path) -> ContinuousSignalData:
        path = Path(path)

        with path.open("r", encoding="latin1", errors="replace") as f:
            lines = f.readlines()

        # ── 1. Extract metadata from ASCII header (lines before image data) ───
        metadata = _extract_header_metadata(lines[:20])

        # ── 2. Find the continuous waveform header ────────────────────────────
        # The file has two "Image\tTime\t..." header rows:
        #   - ~11 columns: breath-averaged Tidal Variations (skip this)
        #   - >20 columns: frame-by-frame waveforms with all Medibus signals (this)
        waveform_header_idx = None
        for i, line in enumerate(lines):
            fields = _split_tab_fields(line)
            if fields and fields[0].lower() == "image" and len(fields) > 20:
                waveform_header_idx = i
                break

        if waveform_header_idx is None:
            raise ValueError(
                "Could not find continuous waveform table in Drager ASC. "
                "Expected a header row starting with 'Image' and >20 columns."
            )

        # ── 3. Read the table with pandas ─────────────────────────────────────
        table_text = "".join(lines[waveform_header_idx:])
        df = pd.read_csv(
            io.StringIO(table_text),
            sep="\t",
            decimal=",",
            na_values=["-"],
            skip_blank_lines=True,
        )

        if df.empty:
            raise ValueError("Malformed Drager ASC: no data rows in continuous waveform table")

        # ── 4. Normalize column names ──────────────────────────────────────────
        df.columns = [_to_snake_case(str(c)) for c in df.columns]

        # ── 5. Fix image index column ──────────────────────────────────────────
        if "image" in df.columns:
            numeric_image = pd.to_numeric(df["image"], errors="coerce")
            df = df[numeric_image.notna()].copy()
            df["image"] = numeric_image[numeric_image.notna()].astype(int).values

        # ── 7. Estimate sampling frequency from time column ────────────────────
        fs = None
        if "time" in df.columns:
            time_values = pd.to_numeric(df["time"], errors="coerce").dropna()
            if len(time_values) >= 2:
                dt = (time_values.diff().dropna()).median()
                if pd.notna(dt) and dt > 0:
                    fs = float(1.0 / dt)

        metadata["parsed_section"] = "continuous_waveforms"
        metadata["n_rows"] = int(len(df))
        metadata["n_columns"] = int(len(df.columns))

        return ContinuousSignalData(
            table=df,
            fs=fs,
            filename=str(path),
            file_format="asc",
            metadata=metadata,
        )
