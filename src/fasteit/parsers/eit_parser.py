"""
Parser for Dräger PulmoVista .eit files.

.eit files contain raw electrode voltages (transimpedances).
Format: ASCII header + binary frame data.
208 transimpedance measurements per frame (16 injections x 13 measures,
adjacent-drive pattern, excluding auto-measurements).

Magic string: "---Draeger EIT-Software---" in first bytes.

NOTE: This is NOT the Carefusion format (which uses block types 3/7/8/10).

TODO (Task 2.1.x): Reverse engineering — hexdump, ASCII strings, header fields
TODO (Task 2.3.1): parse_eit_header(path) -> dict
TODO (Task 2.3.2): autodetect_format(path) -> Literal["draeger", "carefusion"]
TODO (Task 2.4.1): parse_eit_frames(path, header) -> RawData
TODO (Task 2.4.2): compute_transimpedances(frame) -> np.ndarray shape (208,)
"""
