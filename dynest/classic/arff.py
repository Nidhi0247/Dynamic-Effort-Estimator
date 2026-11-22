"""Minimal ARFF reader (text only, no code execution).  Returns one DataFrame per @data block."""
import csv
import io

import numpy as np
import pandas as pd


def read_arff(path):
    names, blocks, cur = [], [], None
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            low = s.lower()
            if low.startswith("@attribute"):
                rest = s[len("@attribute"):].strip()
                if rest.startswith("'"):
                    name = rest[1:rest.index("'", 1)]
                else:
                    name = rest.split()[0]
                names.append(name.strip())
            elif low.startswith("@data"):
                cur = []
                blocks.append(cur)
            elif low.startswith("@"):
                continue
            elif cur is not None:
                cur.append(s)
    out = []
    for rows in blocks:
        recs = list(csv.reader(io.StringIO("\n".join(rows)), quotechar="'", skipinitialspace=True))
        recs = [r for r in recs if len(r) == len(names)]
        df = pd.DataFrame(recs, columns=names).replace({"?": np.nan})
        for c in df:
            num = pd.to_numeric(df[c], errors="coerce")
            if num.notna().sum() >= 0.9 * df[c].notna().sum() and df[c].notna().any():
                df[c] = num
        out.append(df)
    return out
