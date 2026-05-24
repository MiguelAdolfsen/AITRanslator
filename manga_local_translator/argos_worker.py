from __future__ import annotations

import json
import sys


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    if len(args) != 2:
        print(json.dumps({"ok": False, "error": "Expected source and target language codes"}))
        return 2

    source_code, target_code = args
    payload = json.loads(sys.stdin.read())
    text = str(payload.get("text", ""))

    try:
        import argostranslate.translate

        installed_languages = argostranslate.translate.get_installed_languages()
        source = next((lang for lang in installed_languages if lang.code == source_code), None)
        target = next((lang for lang in installed_languages if lang.code == target_code), None)
        if source is None or target is None:
            raise RuntimeError(f"Missing Argos language model {source_code}->{target_code}")
        translation = source.get_translation(target)
        if translation is None:
            raise RuntimeError(f"Missing Argos translation {source_code}->{target_code}")
        print(json.dumps({"ok": True, "text": translation.translate(text)}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": repr(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

