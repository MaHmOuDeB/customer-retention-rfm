"""Download and unpack UCI Online Retail II into data/ (about 45 MB).  python src/download.py"""
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
DATA = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    target = DATA / "online_retail_II.xlsx"
    if target.exists():
        print(f"{target.name} already there")
        return
    DATA.mkdir(exist_ok=True)
    archive = DATA / "online_retail_ii.zip"
    print("downloading", URL)
    urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(DATA)
    archive.unlink()
    print("saved", target)


if __name__ == "__main__":
    main()
