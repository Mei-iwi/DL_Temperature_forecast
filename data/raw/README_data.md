# Raw Temperature Data

Place the original temperature CSV in this folder as:

```text
data/raw/temperature.csv
```

Supported input formats:

- NASA POWER daily CSV with metadata header and columns `YEAR,DOY,T2M`.
- NASA POWER daily CSV with columns `YEAR,MO,DY,T2M`.
- Plain CSV with columns `date,temperature`.

Rules:

- Keep the raw file unchanged.
- NASA missing value marker `-999` is treated as missing data.
- The raw CSV is ignored by Git; only this README and `.gitkeep` are tracked.
