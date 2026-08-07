from pathlib import Path
from hashlib import sha256

def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def stable_filename(path: Path, ext: str = "") -> str:
    """Returns a filename built from the original stem + its SHA-256 hash."""
    digest = file_sha256(path)[:16]
    suffix = ext or path.suffix
    return f"{path.stem}_{digest}{suffix}"