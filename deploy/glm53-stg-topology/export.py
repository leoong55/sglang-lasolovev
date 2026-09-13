"""Export a named experiment directory as small independently verifiable log chunks."""

import argparse
import base64
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/results")
    p.add_argument("--run-id", required=True)
    p.add_argument(
        "--part",
        type=int,
        default=-1,
        help="-1 prints the manifest; otherwise a zero-based part",
    )
    args = p.parse_args()
    root = Path(args.root).resolve()
    src = (root / args.run_id).resolve()
    if not src.is_relative_to(root) or src == root or not src.is_dir():
        p.error("run-id must identify one existing directory below results root")
    # Export is read-only except for an ephemeral local tar file. The exact sorted
    # member metadata is preserved, so sequential invocations give identical bytes.
    with tempfile.TemporaryFile() as tmp:
        import gzip

        with gzip.GzipFile(fileobj=tmp, mode="wb", mtime=0, filename="") as zipped:
            with tarfile.open(fileobj=zipped, mode="w|") as tar:
                for path in sorted(src.rglob("*")):
                    if path.is_symlink():
                        raise ValueError("Refusing symlink in results: " + str(path))
                    if path.is_file():
                        info = tar.gettarinfo(
                            str(path), arcname=str(path.relative_to(root))
                        )
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        with path.open("rb") as f:
                            tar.addfile(info, f)
        tmp.seek(0)
        # One base64 record stays below 1.4 MiB, well below normal kubelet log
        # rotation, while avoiding a separate CPU Job for every 256 KiB.
        chunk_size = 1024 * 1024
        parts = []
        digest = hashlib.sha256()
        while raw := tmp.read(chunk_size):
            digest.update(raw)
            parts.append(hashlib.sha256(raw).hexdigest())
        manifest = {
            "run_id": args.run_id,
            "part_bytes": chunk_size,
            "parts": parts,
            "sha256": digest.hexdigest(),
            "bytes": tmp.tell(),
        }
        if args.part < 0:
            print("GLM53_EXPORT_MANIFEST " + json.dumps(manifest), flush=True)
        else:
            if args.part >= len(parts):
                p.error("part index out of range")
            tmp.seek(args.part * chunk_size)
            raw = tmp.read(chunk_size)
            print(
                "GLM53_EXPORT_PART "
                + json.dumps(
                    {
                        "index": args.part,
                        "sha256": parts[args.part],
                        "data": base64.b64encode(raw).decode(),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
