from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scaffold a draft real benchmark case from compact debug artifacts.")
    parser.add_argument("--debug-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-name", required=True)
    args = parser.parse_args(argv)
    if not args.debug_dir.exists():
        raise SystemExit(f"debug dir does not exist: {args.debug_dir}")
    target = args.output / "cases" / args.case_name
    for name in ("pages", "labels", "ignore_regions", "configs"):
        (target / name).mkdir(parents=True, exist_ok=True)
    copied = 0
    for image in sorted(args.debug_dir.rglob("*.png")):
        page_id = f"page_{copied + 1:03d}"
        shutil.copy2(image, target / "pages" / f"{page_id}.png")
        label = {
            "schema_version": 1,
            "page_id": page_id,
            "image": f"pages/{page_id}.png",
            "text_regions": [],
            "groups": [],
            "notes": f"Draft label scaffolded from {image}. Human review required before use as benchmark ground truth.",
        }
        (target / "labels" / f"{page_id}.labels.json").write_text(
            json.dumps(label, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        copied += 1
    (target / "README.md").write_text(
        f"# {args.case_name}\n\nDraft benchmark scaffold. Human review required before scoring.\n",
        encoding="utf-8",
    )
    print(f"Scaffolded {copied} draft page(s) in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

