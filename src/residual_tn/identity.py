"""Content identities and atomic metadata for safe checkpoint reuse."""

from pathlib import Path
import hashlib
import json
import os


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity():
    root = Path(__file__).resolve().parent
    entries = {
        str(p.relative_to(root)): sha256(p)
        for p in sorted(root.rglob("*"))
        if p.suffix in (".py", ".json")
    }
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def claim_run(directory, identity):
    """Reject incompatible outputs before any expensive work."""
    path = Path(directory) / "run_identity.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError(
                "Output contains incompatible model/code checkpoints; use a new output directory"
            )
    else:
        atomic_json(path, identity)


class RunLock:
    """A kernel-managed lock prevents concurrent writers; crashes release it."""

    def __init__(self, directory):
        self.directory = Path(directory)

    def __enter__(self):
        import fcntl

        self.directory.mkdir(parents=True, exist_ok=True)
        self.stream = (self.directory / ".run.lock").open("a+")
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            raise RuntimeError("Another process owns this output directory") from None
        self.stream.seek(0)
        self.stream.truncate()
        self.stream.write(str(os.getpid()))
        self.stream.flush()
        return self

    def __exit__(self, *args):
        self.stream.close()
