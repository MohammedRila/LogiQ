from dataclasses import dataclass
from typing import Optional, Dict
import re
from ..utils.text import parse_basic_fields, templateize, normalize_whitespace

@dataclass
class LogRecord:
    raw: str
    timestamp: Optional[str]
    level: Optional[str]
    message: str
    template: str
    structured_fields: Dict[str, str]  # New: extracted fields like IP, user ID, port

# Standard log levels
LOG_LEVELS = {"DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL", "FATAL"}

# Precompiled regex patterns for efficiency
IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
USER_PATTERN = re.compile(r"\buser(?:_id)?[:=]?(\d+)\b", re.IGNORECASE)
PORT_PATTERN = re.compile(r"\bport[:=]?(\d+)\b", re.IGNORECASE)
ERROR_CODE_PATTERN = re.compile(r"\bcode[:=]?(\d{3,5})\b", re.IGNORECASE)

def extract_structured_fields(message: str) -> Dict[str, str]:
    """Extract structured fields like IP, user ID, port, and error code from a log message."""
    fields = {}
    ip_match = IP_PATTERN.findall(message)
    if ip_match:
        fields["ip"] = ip_match[0]  # Take first match
    
    user_match = USER_PATTERN.findall(message)
    if user_match:
        fields["user_id"] = user_match[0]
    
    port_match = PORT_PATTERN.findall(message)
    if port_match:
        fields["port"] = port_match[0]
    
    error_code_match = ERROR_CODE_PATTERN.findall(message)
    if error_code_match:
        fields["error_code"] = error_code_match[0]
    
    return fields

def parse_line(line: str) -> LogRecord:
    """
    Parse a single log line into a structured LogRecord.

    - Extracts timestamp and log level via parse_basic_fields()
    - Normalizes whitespace
    - Generates a template string
    - Extracts structured fields like IPs, user IDs, ports, and error codes
    """
    # Step 1: Extract basic fields
    fields = parse_basic_fields(line)
    
    # Step 2: Normalize message
    message = normalize_whitespace(line)
    
    # Step 3: Standardize log level
    level = fields.get("level")
    if level:
        level = level.upper()
        if level not in LOG_LEVELS:
            level = None
    
    # Step 4: Generate template
    template = templateize(message)
    
    # Step 5: Extract structured fields
    structured_fields = extract_structured_fields(message)
    
    return LogRecord(
        raw=line,
        timestamp=fields.get("timestamp"),
        level=level,
        message=message,
        template=template,
        structured_fields=structured_fields
    )

def parse_lines(lines):
    """Parse multiple log lines into LogRecord objects."""
    return [parse_line(line) for line in lines if line.strip()]
