"""
Parser for Dräger PulmoVista .txt / .csv export files.

PulmoVista exports a breath-level summary CSV with columns in German/English
depending on firmware version. Separator is semicolon. Encoding varies.

TODO (Task 3.2.1): Document CSV format variants (separator, header, units)
TODO (Task 3.3.1): parse_csv(path, config) -> pd.DataFrame
TODO (Task 3.3.2): autodetect_encoding(path) -> str  (UTF-8 vs Latin-1)
"""
