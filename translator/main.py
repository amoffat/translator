import argparse
import json
import os
import pathlib
import time
from datetime import timedelta
import dotenv

import openai
from tqdm import tqdm

from . import llm
from .langs import SUPPORTED_LANGS


def locale_dir(*, trans_dir, lang) -> pathlib.Path:
    return trans_dir / lang


def translation_path(*, trans_dir, lang, ns):
    return locale_dir(trans_dir=trans_dir, lang=lang) / f"{ns}.jsonl"


def load_translations(*, trans_dir, lang, ns) -> dict[str, dict]:
    path = translation_path(trans_dir=trans_dir, lang=lang, ns=ns)
    translations = {}
    try:
        with open(path, "r", encoding="utf8") as h:
            for line in h:
                try:
                    obj = json.loads(line)
                    translations[obj["k"]] = obj
                except json.JSONDecodeError:
                    continue
        return translations
    except FileNotFoundError:
        return {}


def write_translations(*, trans_dir, lang, ns, translations: dict[str, dict]) -> None:
    d = locale_dir(trans_dir=trans_dir, lang=lang)
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)

    path = translation_path(trans_dir=trans_dir, lang=lang, ns=ns)
    with open(path, "w", encoding="utf8") as h:
        keys = sorted(translations.keys())
        for key in keys:
            obj = translations[key]
            h.write(json.dumps(obj, ensure_ascii=False) + "\n")


def source_files(dir):
    for file in dir.glob("*.jsonl"):
        yield file


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate text")
    parser.add_argument(
        "--trans-dir",
        type=pathlib.Path,
        required=True,
        help="Directory for translations",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-nano",
        help="LLM model to use for translation",
    )

    parser.add_argument(
        "--dotenv",
        type=pathlib.Path,
        help="Load environment variables from .env file",
    )

    args = parser.parse_args()
    trans_dir = args.trans_dir

    if args.dotenv:
        dotenv.load_dotenv(args.dotenv)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set in .env")

    client = llm.Client(
        openai_client=openai.OpenAI(api_key=api_key),
        model=args.model,
    )

    entries: dict[str, dict] | None = None
    total_entries = 0
    num_trans_langs = len(SUPPORTED_LANGS)
    for source in source_files(trans_dir / "main"):
        entries = load_translations(
            trans_dir=trans_dir,
            lang="main",
            ns=source.stem,
        )
        total_entries += len(entries)

    assert entries, "No source translations found"
    total_translations = total_entries * num_trans_langs

    # detect primary language
    sample = "\n".join([obj["v"] for obj in list(entries.values())[:10]][:500])
    primary_lang = llm.detect_lang(client=client, text=sample)
    assert primary_lang, "Failed to detect primary language"
    tqdm.write(f"Detected language: {primary_lang}")

    processed = 0
    start_time = time.time()
    with tqdm(
        total=total_translations,
        ncols=80,
        dynamic_ncols=True,
        desc="Translating",
        unit="entry",
    ) as pbar:
        for source in source_files(trans_dir / "main"):
            ns = source.stem
            entries = load_translations(
                trans_dir=trans_dir,
                lang="main",
                ns=ns,
            )

            for lang_code, lang_name in SUPPORTED_LANGS.items():
                lang_translations = load_translations(
                    trans_dir=trans_dir,
                    lang=lang_code,
                    ns=ns,
                )

                # cleanup old translations
                old_keys = set(lang_translations.keys())
                active_keys = set(entries.keys())
                to_remove = old_keys - active_keys
                for key in to_remove:
                    lang_translations.pop(key, None)

                is_primary_lang = lang_code == primary_lang

                tqdm.write(f"Translating to {lang_name} ({lang_code})...")

                def write():
                    write_translations(
                        trans_dir=trans_dir,
                        lang=lang_code,
                        ns=ns,
                        translations=lang_translations,
                    )

                for source_entry in entries.values():
                    key = source_entry["k"]
                    to_translate = source_entry["v"]
                    context = source_entry.get("ctx")
                    trans_entry = lang_translations.setdefault(key, {})

                    locked = trans_entry.get("lock", False)

                    cur_trans = None
                    if is_primary_lang:
                        cur_trans = to_translate
                    elif trans_entry:
                        cur_trans = trans_entry.get("v")

                    if not cur_trans:
                        needs_translation = True
                    # If primary language, no translation needed
                    elif is_primary_lang:
                        needs_translation = False
                    else:
                        same_ctx = context == trans_entry.get("ctx")
                        same_trans = to_translate == trans_entry.get("original")
                        inputs_match = same_ctx and same_trans
                        needs_translation = not inputs_match

                    if needs_translation and not locked:
                        try:
                            time.sleep(0.5)
                            updated_translation = llm.translate(
                                client=client,
                                context=context,
                                dest_code=lang_code,
                                to_translate=to_translate,
                            )
                            assert updated_translation, "Translation failed"
                        except KeyboardInterrupt:
                            write()
                            tqdm.write("Translation interrupted by user, quitting")
                            exit()
                        else:
                            tqdm.write(
                                f'    Translated `{key}` to "{updated_translation}"'
                            )
                    else:
                        updated_translation = cur_trans

                    lang_translations[key] = {
                        "k": key,
                        "v": updated_translation,
                        "original": to_translate,
                        "ctx": context,
                        "lock": locked,
                    }
                    processed += 1

                    write()

                    # Progress bar update and ETA
                    elapsed = time.time() - start_time
                    if processed > 0:
                        rate = elapsed / processed
                        remaining = total_translations - processed
                        eta = timedelta(seconds=int(rate * remaining))
                        pbar.set_postfix({"ETA": str(eta)})
                    pbar.update(1)


if __name__ == "__main__":
    main()
