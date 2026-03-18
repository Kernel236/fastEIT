"""
DataFrame assembly and export.

Assembles one row per breath (or per PEEP step) from a Session,
in tidyverse-compatible snake_case format.

TODO (Task 6.1.1): session_to_dataframe(session) -> pd.DataFrame
TODO (Task 6.1.2): Full column list per reference doc section 4.1
TODO (Task 6.2.1): save_hdf5(session, path) — frames, tidal_images, lung_mask
TODO (Task 6.4.1): to_csv(df, path)
TODO (Task 6.4.2): to_parquet(df, path)
TODO (Task 6.4.3): to_excel(df, path)  [optional, openpyxl]
TODO (Task 6.4.4): to_peep_step_csv(session, path) — 1 row per step
"""
