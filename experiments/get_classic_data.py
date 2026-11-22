"""Download the public classic effort datasets used by E8 into data/classic_raw (text ARFF files only)."""
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "classic_raw"
DASE = "https://raw.githubusercontent.com/danrodgar/DASE/master/datasets/effortEstimation/"
FILES = {
    "albrecht_full.arff": "https://zenodo.org/api/records/268467/files/albrecht.arff/content",   # PROMISE, Zenodo
    "kemerer_full.arff": "https://zenodo.org/api/records/268464/files/kemerer.arff/content",
    **{f: DASE + f for f in ["china.arff", "coc81_1_1.arff", "nasa93.arff", "desharnais.arff", "maxwell.arff",
                             "kitchenham.arff", "miyazaki94.arff"]},
}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dst = RAW / name
        if dst.exists():
            print("have", name)
            continue
        with urllib.request.urlopen(url, timeout=60) as r:
            dst.write_bytes(r.read())
        print("got ", name, dst.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
