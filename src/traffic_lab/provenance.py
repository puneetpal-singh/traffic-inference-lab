"""Artifact identity helpers with no model/runtime imports."""

from hashlib import sha256
from pathlib import Path


def file_hash(path):
    path = Path(path)
    if path.is_dir():
        digest = sha256()
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(child.relative_to(path).as_posix().encode())
            digest.update(bytes.fromhex(file_hash(child)))
        return digest.hexdigest()
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
