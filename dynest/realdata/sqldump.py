"""Stream a mysqldump file into one parquet file per table, without needing a MySQL server.

Usage:  python -m dynest.realdata.sqldump data/raw/TAWOS.sql data/tawos
"""
import re
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_CREATE = re.compile(r"^CREATE TABLE `(\w+)`")
_COL = re.compile(r"^\s+`(\w+)` (\w+)")
_INSERT = re.compile(r"^INSERT INTO `(\w+)` VALUES ")
_TOKEN = re.compile(r"\(|\)|'((?:[^'\\]|\\.)*)'|(NULL)|(-?[0-9][0-9.eE+-]*)|,", re.S)
_ESC = {"0": "\0", "n": "\n", "r": "\r", "t": "\t", "Z": "\x1a", "b": "\b"}
_ESC_RE = re.compile(r"\\(.)", re.S)


def _unescape(s):
    return _ESC_RE.sub(lambda m: _ESC.get(m.group(1), m.group(1)), s) if "\\" in s else s


def _arrow_type(sql_type):
    t = sql_type.lower()
    if t in ("int", "tinyint", "smallint", "bigint", "mediumint"):
        return pa.int64()
    if t in ("double", "float", "decimal"):
        return pa.float64()
    if t in ("datetime", "timestamp", "date"):
        return pa.timestamp("s")
    return pa.string()


def _convert(value, typ):
    if value is None:
        return None
    if pa.types.is_integer(typ):
        return int(float(value))
    if pa.types.is_floating(typ):
        return float(value)
    if pa.types.is_timestamp(typ) and value.startswith("0000"):
        return None
    return value


class _TableWriter:
    def __init__(self, path, cols, types, flush_rows=200_000):
        self.path, self.cols, self.types, self.flush_rows = path, cols, types, flush_rows
        self.buf = [[] for _ in cols]
        self.writer = None
        self.rows = 0

    def add(self, row):
        if len(row) != len(self.cols):
            raise ValueError(f"{self.path.stem}: expected {len(self.cols)} values, got {len(row)}")
        for b, v, t in zip(self.buf, row, self.types):
            b.append(_convert(v, t))
        self.rows += 1
        if len(self.buf[0]) >= self.flush_rows:
            self.flush()

    def flush(self):
        if not self.buf[0]:
            return
        arrays = [pa.array(b, pa.string()).cast(t) if pa.types.is_timestamp(t) else pa.array(b, t)
                  for b, t in zip(self.buf, self.types)]
        table = pa.Table.from_arrays(arrays, names=self.cols)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table)
        self.buf = [[] for _ in self.cols]

    def close(self):
        self.flush()
        if self.writer is not None:
            self.writer.close()


def _parse_values(body, emit):
    row = []
    for m in _TOKEN.finditer(body):
        tok = m.group(0)
        if tok == "(":
            row = []
        elif tok == ")":
            emit(row)
        elif tok == ",":
            continue
        elif tok[0] == "'":
            row.append(_unescape(m.group(1)))
        elif m.group(2):
            row.append(None)
        else:
            row.append(m.group(3))


def convert(dump_path, out_dir, tables=None, log=print):
    dump_path, out_dir = Path(dump_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schemas, current, writers, logged = {}, None, {}, {}
    t0, done_bytes, total = time.time(), 0, dump_path.stat().st_size
    with open(dump_path, "r", encoding="utf-8", errors="replace", newline="\n") as fh:
        for line in fh:
            done_bytes += len(line)
            m = _CREATE.match(line)
            if m:
                current = m.group(1)
                schemas[current] = []
                continue
            if current:
                if line.startswith(")"):
                    current = None
                else:
                    c = _COL.match(line)
                    if c:
                        schemas[current].append((c.group(1), _arrow_type(c.group(2))))
                continue
            m = _INSERT.match(line)
            if not m:
                continue
            name = m.group(1)
            if tables and name not in tables:
                continue
            if name not in writers:
                writers[name] = _TableWriter(out_dir / f"{name}.parquet",
                                             [c for c, _ in schemas[name]], [t for _, t in schemas[name]])
                logged[name] = 0
                log(f"  -> {name}")
            _parse_values(line[m.end():].rstrip().rstrip(";"), writers[name].add)
            if writers[name].rows - logged[name] >= 1_000_000:
                logged[name] = writers[name].rows
                log(f"     {name}: {logged[name]:,} rows | {100 * done_bytes / total:5.1f}% | {time.time() - t0:6.0f}s")
    counts = {}
    for name, w in writers.items():
        w.close()
        counts[name] = w.rows
        log(f"  done {name}: {w.rows:,} rows")
    return counts


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/raw/TAWOS.sql"
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/tawos"
    convert(src, dst)
