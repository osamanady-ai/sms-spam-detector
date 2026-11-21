"""Unpack bundled real datasets/reports safely on first use; no network requests."""
from pathlib import Path
import hashlib
import zipfile

BASE=Path(__file__).resolve().parent
EXPECTED_SHA256='2d8d70ab4eabe8d48771a0a306c68b6abd411c358bbda2b4df8e11bd9df0f474'


def ensure_assets():
    if (BASE/'.assets-ready').exists():
        return
    candidates=list(BASE.glob('*project-assets.zip'))
    if len(candidates)!=1:
        raise FileNotFoundError('Keep the bundled project-assets.zip archive beside main.py. Its numeric upload prefix may vary.')
    archive=candidates[0]
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=EXPECTED_SHA256:
        raise ValueError('Bundled dataset archive checksum does not match; download a fresh repository copy.')
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target=(BASE/item.filename).resolve()
            if not target.is_relative_to(BASE):
                raise ValueError('Unsafe archive path')
        z.extractall(BASE)
    (BASE/'.assets-ready').write_text(EXPECTED_SHA256)


if __name__=='__main__':
    ensure_assets()
    print('Bundled datasets, reports and chart assets are ready.')
