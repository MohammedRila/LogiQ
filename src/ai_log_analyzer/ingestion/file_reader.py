from pathlib import Path
from typing import Iterable
import time

def read_lines(path: str, live: bool = False, sleep_sec: float = 1.0) -> Iterable[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found")
    if p.is_file():
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            if live:
                f.seek(0, 2)  # move to EOF
                while True:
                    line = f.readline()
                    if line:
                        yield line.strip()
                    else:
                        time.sleep(sleep_sec)
            else:
                for line in f:
                    yield line.strip()
    elif p.is_dir():
        for fp in p.rglob("*"):
            if fp.is_file():
                with fp.open("r", encoding="utf-8", errors="ignore") as f:
                    if live:
                        f.seek(0, 2)
                        while True:
                            line = f.readline()
                            if line:
                                yield line.strip()
                            else:
                                time.sleep(sleep_sec)
                    else:
                        for line in f:
                            yield line.strip()
