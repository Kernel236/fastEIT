# Dräger PulmoVista EIT File Format Specification

> **Status**: Work in progress — to be completed in Task 2.8.1
>
> This document is intended as a standalone publishable specification
> of the Dräger PulmoVista .eit file format, independent of any GPL code.

---

## Overview

The Dräger PulmoVista 500 generates three file types per recording:

| Extension | Content | Use |
|-----------|---------|-----|
| `.eit` | Raw electrode voltages (transimpedances) | Offline reconstruction |
| `.bin` | Reconstructed 32×32 impedance images | Direct clinical use |
| `.txt` | Breath-level summary (CSV-like) | Quick analysis |

---

## .eit Format

**To be documented in Task 2.1.x (reverse engineering) and Task 2.8.1**

Known facts:
- Header: ASCII text
- Magic string: `---Draeger EIT-Software---` (to confirm)
- Data section: binary, little-endian
- 208 transimpedance measurements per frame
- 16 electrodes, adjacent-drive stimulation pattern, 5 mA injection current
- Measurement pattern: `rotate_meas` (208 = 16 × 13, excluding auto-measurements)

**NOT** the Carefusion format (which uses block types 3, 7, 8, 10).

---

## .bin Format

**To be documented in Task 1.x**

Known facts:
- Frame size: 4358 bytes (base) or 4382 bytes (extended with Medibus data)
- 32×32 = 1024 float32 pixels per frame
- Little-endian (`<f4`)
- Sample rate: 20 Hz (standard PulmoVista setting)

---

## References

- EIDORS source: `eidors_readdata.m`, function `read_draeger_header` (GPL — format study only)
- See `docs/eidors_analysis.md` for annotated EIDORS findings
