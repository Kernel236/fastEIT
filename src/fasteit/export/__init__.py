"""Layer 4: the results of layer 3 as tables for statistical analysis and databases.

```python
from fasteit.export import combine_tables, recording_tables, write_csv

tables = combine_tables([
    recording_tables(baseline, phase="baseline"),
    recording_tables(trial, phase="trial"),
])
write_csv(tables, "results/")
```
"""

from fasteit.export.tables import KEYS, combine_tables, recording_tables, write_csv

__all__ = ["KEYS", "combine_tables", "recording_tables", "write_csv"]
