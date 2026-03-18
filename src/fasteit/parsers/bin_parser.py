"""
Parser for Dräger PulmoVista .bin files.

.bin files contain reconstructed 32x32 impedance images at 20 Hz,
one frame per time step. Frame size is either 4358 or 4382 bytes
depending on firmware version (with/without extended Medibus data).

TODO (Task 1.1.1): autodetect_frame_size(file_size) -> int
TODO (Task 1.2.1): Define numpy structured dtype for a single frame
TODO (Task 1.2.2): parse_bin(path) -> RawData  (fast, np.fromfile)
TODO (Task 1.2.3): parse_bin_stream(path) -> Iterator[Frame]  (>500 MB)
TODO (Task 1.3.1): Investigate 208-byte padding field (Medibus?)
TODO (Task 1.3.3): extract_medibus(frame) -> dict
TODO (Task 1.3.4): extract_pressure_pod(frame) -> dict
"""
