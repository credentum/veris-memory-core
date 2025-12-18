"""
Write-Ahead Log (WAL) Shadow

Provides tamper-evident logging by writing audit entries to an
append-only log file before they're committed to Qdrant.

"A memory that can be rewritten isn't memory—it's fiction."

The WAL serves as:
1. Crash recovery - replay uncommitted entries
2. Tamper evidence - compare WAL against Qdrant
3. External audit - ship WAL to immutable storage (S3)

Format: JSONL (one JSON object per line)
Each line includes the entry plus a line hash for chain verification.
"""

import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from loguru import logger

from .models import AuditEntry


class WriteAheadLog:
    """
    Append-only write-ahead log for audit entries.

    File format (JSONL):
    {"seq": 1, "line_hash": "abc123...", "prev_line_hash": null, "entry": {...}}
    {"seq": 2, "line_hash": "def456...", "prev_line_hash": "abc123...", "entry": {...}}

    Features:
    - Append-only (file opened in append mode with exclusive lock)
    - Line-level hash chain (each line hashes the previous)
    - Crash-safe (fsync after each write)
    - Rotatable (close and archive, start new file)
    """

    def __init__(
        self,
        log_dir: str = "/var/log/veris/audit",
        log_prefix: str = "audit_wal",
        max_file_size_mb: int = 100,
        sync_on_write: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.log_prefix = log_prefix
        self.max_file_size_mb = max_file_size_mb
        self.sync_on_write = sync_on_write

        self._current_file: Optional[Path] = None
        self._file_handle = None
        self._sequence_number = 0
        self._prev_line_hash: Optional[str] = None
        self._write_count = 0
        self._bytes_written = 0

        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize or resume from existing WAL
        self._initialize()

    def _initialize(self) -> None:
        """Initialize WAL, resuming from existing file if present."""
        # Find the latest WAL file
        existing_files = sorted(self.log_dir.glob(f"{self.log_prefix}_*.jsonl"))

        if existing_files:
            self._current_file = existing_files[-1]
            self._resume_from_file(self._current_file)
        else:
            self._start_new_file()

    def _start_new_file(self) -> None:
        """Start a new WAL file."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.log_prefix}_{timestamp}.jsonl"
        self._current_file = self.log_dir / filename

        self._close_current_file()
        self._file_handle = open(self._current_file, "a")
        # Acquire exclusive lock immediately after opening
        fcntl.flock(self._file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self._sequence_number = 0
        self._prev_line_hash = None

        logger.info(f"Started new WAL file: {self._current_file}")

    def _resume_from_file(self, filepath: Path) -> None:
        """Resume from an existing WAL file."""
        logger.info(f"Resuming WAL from: {filepath}")

        # Read the last line to get sequence and hash
        last_line = None
        try:
            with open(filepath, "r") as f:
                for line in f:
                    if line.strip():
                        last_line = line.strip()
        except Exception as e:
            logger.warning(f"Error reading WAL file: {e}")
            self._start_new_file()
            return

        if last_line:
            try:
                data = json.loads(last_line)
                self._sequence_number = data["seq"]
                self._prev_line_hash = data["line_hash"]
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error parsing last WAL line: {e}")
                self._sequence_number = 0
                self._prev_line_hash = None

        self._file_handle = open(filepath, "a")
        # Acquire exclusive lock immediately after opening
        fcntl.flock(self._file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self._bytes_written = filepath.stat().st_size

    def _close_current_file(self) -> None:
        """Close current file handle and release lock."""
        if self._file_handle:
            try:
                self._file_handle.flush()
                os.fsync(self._file_handle.fileno())
                # Release exclusive lock before closing
                fcntl.flock(self._file_handle.fileno(), fcntl.LOCK_UN)
                self._file_handle.close()
            except Exception as e:
                logger.warning(f"Error closing WAL file: {e}")
            finally:
                self._file_handle = None

    def _should_rotate(self) -> bool:
        """Check if file should be rotated."""
        if self._bytes_written > self.max_file_size_mb * 1024 * 1024:
            return True
        return False

    def _compute_line_hash(self, data: dict) -> str:
        """Compute hash for a WAL line."""
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def append(self, entry: AuditEntry) -> dict:
        """
        Append an audit entry to the WAL.

        Returns the WAL record (with seq, line_hash, etc.).
        """
        if self._should_rotate():
            self._start_new_file()

        self._sequence_number += 1

        # Build the WAL record
        record = {
            "seq": self._sequence_number,
            "prev_line_hash": self._prev_line_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "entry": entry.model_dump(mode="json"),
        }

        # Compute line hash (includes prev_line_hash for chaining)
        record["line_hash"] = self._compute_line_hash(record)
        self._prev_line_hash = record["line_hash"]

        # Write to file with exclusive lock
        line = json.dumps(record, separators=(",", ":")) + "\n"
        line_bytes = line.encode()

        try:
            # Acquire exclusive lock for write
            fcntl.flock(self._file_handle.fileno(), fcntl.LOCK_EX)
            try:
                self._file_handle.write(line)
                if self.sync_on_write:
                    self._file_handle.flush()
                    os.fsync(self._file_handle.fileno())
                self._bytes_written += len(line_bytes)
                self._write_count += 1
            finally:
                fcntl.flock(self._file_handle.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            logger.exception(f"WAL write failed: {e}")
            raise

        return record

    def flush(self) -> None:
        """Flush the current file to disk."""
        if self._file_handle:
            self._file_handle.flush()
            os.fsync(self._file_handle.fileno())

    def verify_chain(self, filepath: Optional[Path] = None) -> dict:
        """
        Verify the hash chain integrity of a WAL file.

        Returns verification result with any broken links.
        """
        # Flush pending writes before verification
        if filepath is None:
            self.flush()

        filepath = filepath or self._current_file
        if not filepath or not filepath.exists():
            return {"valid": False, "error": "File not found"}

        broken_links = []
        line_count = 0
        prev_hash = None

        with open(filepath, "r") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    broken_links.append({
                        "line": line_num,
                        "error": f"JSON decode error: {e}",
                    })
                    continue

                # Verify prev_line_hash matches
                if data.get("prev_line_hash") != prev_hash:
                    broken_links.append({
                        "line": line_num,
                        "seq": data.get("seq"),
                        "error": "prev_line_hash mismatch",
                        "expected": prev_hash,
                        "actual": data.get("prev_line_hash"),
                    })

                # Verify line_hash is correct
                stored_hash = data.pop("line_hash", None)
                computed_hash = self._compute_line_hash(data)
                if stored_hash != computed_hash:
                    broken_links.append({
                        "line": line_num,
                        "seq": data.get("seq"),
                        "error": "line_hash mismatch (tampering detected)",
                        "stored": stored_hash[:16] if stored_hash else None,
                        "computed": computed_hash[:16],
                    })

                prev_hash = stored_hash
                line_count += 1

        return {
            "valid": len(broken_links) == 0,
            "file": str(filepath),
            "line_count": line_count,
            "broken_links": broken_links,
        }

    def iterate_entries(
        self,
        filepath: Optional[Path] = None,
        start_seq: int = 0,
    ) -> Iterator[dict]:
        """Iterate over WAL entries, optionally from a starting sequence."""
        # Flush pending writes before iteration
        if filepath is None:
            self.flush()

        filepath = filepath or self._current_file
        if not filepath or not filepath.exists():
            return

        with open(filepath, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line.strip())
                    if data.get("seq", 0) >= start_seq:
                        yield data
                except json.JSONDecodeError:
                    continue

    def get_stats(self) -> dict:
        """Get WAL statistics."""
        return {
            "current_file": str(self._current_file) if self._current_file else None,
            "sequence_number": self._sequence_number,
            "write_count": self._write_count,
            "bytes_written": self._bytes_written,
            "prev_line_hash": self._prev_line_hash[:16] if self._prev_line_hash else None,
        }

    def close(self):
        """Close the WAL cleanly."""
        self._close_current_file()
        logger.info(
            f"WAL closed",
            writes=self._write_count,
            bytes=self._bytes_written,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
