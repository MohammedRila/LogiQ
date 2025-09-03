
import re
from typing import Dict, Optional

LEVELS = ["TRACE","DEBUG","INFO","WARN","WARNING","ERROR","CRITICAL","FATAL"]

TIMESTAMP_RE = re.compile(
    r"""(?P<ts>
        \d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z? |
        [A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2} |
        \d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}
    )""",
    re.VERBOSE,
)

LEVEL_RE = re.compile(r"\b(?P<lvl>TRACE|DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL|FATAL)\b")

HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
NUM_RE = re.compile(r"\b\d+\b")
UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b")

def parse_basic_fields(line: str) -> Dict[str, Optional[str]]:
    ts = None
    m = TIMESTAMP_RE.search(line)
    if m:
        ts = m.group("ts")
    lvl = None
    m = LEVEL_RE.search(line)
    if m:
        lvl = m.group("lvl")
    return {"timestamp": ts, "level": lvl}

def templateize(msg: str) -> str:
    msg = UUID_RE.sub("<UUID>", msg)
    msg = IP_RE.sub("<IP>", msg)
    msg = HEX_RE.sub("<HEX>", msg)
    msg = NUM_RE.sub("<NUM>", msg)
    return msg

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()
