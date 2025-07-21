import argparse
import hashlib
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
    return locale_dir(trans_dir=trans_dir, lang=lang) / f"{ns}.json"


def load_translations(*, trans_dir, lang, ns):
    path = translation_path(trans_dir=trans_dir, lang=lang, ns=ns)
    try:
        with open(path, "r", encoding="utf8") as h:
            return json.load(h)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_translations(*, trans_dir, lang, ns, translations):
    d = locale_dir(trans_dir=trans_dir, lang=lang)
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)

    with open(
        translation_path(
            trans_dir=trans_dir,
            lang=lang,
            ns=ns,
        ),
        "w",
        encoding="utf8",
    ) as h:
        json.dump(translations, h, ensure_ascii=False, indent=2)


def load_source(source_trans):
    with open(source_trans, "r", encoding="utf8") as h:
        raw_entries = json.load(h)

    entries = []
    for key, value in raw_entries.items():
        if not key.endswith("_"):
            ctx = raw_entries.get(key + "_")
            entries.append((key, value, ctx))

    return entries


def h(text):
    return hashlib.sha1(text.encode()).hexdigest()


def source_entries(dir):
    for file in dir.glob("*.json"):
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
        "--dotenv", type=pathlib.Path,
        help="Load environment variables from .env file",
    )

    args = parser.parse_args()

    trans_dir = args.trans_dir
    cache_file = trans_dir / "cache.json"
    if cache_file.exists():
        cache = json.load(open(cache_file, "r"))
    else:
        cache = {}

    def sync_cache():
        with open(cache_file, "w") as h:
            json.dump(cache, h, indent=2)

    if args.dotenv:
        dotenv.load_dotenv(args.dotenv)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set in .env")

    client = llm.Client(
        openai_client=openai.OpenAI(api_key=api_key),
        model=args.model,
    )

    total_entries = 0
    num_trans_langs = len(SUPPORTED_LANGS)
    for source in source_entries(trans_dir / "main"):
        entries = load_source(source)
        total_entries += len(entries)

    total_translations = total_entries * num_trans_langs

    primary_lang: str | None = cache.get("lang")

    source = next(source_entries(trans_dir / "main"))
    if not primary_lang:
        entries = load_source(next(source_entries(trans_dir / "main")))
        sample = "\n".join([value for _, value, _ in entries[:10]][:500])
        primary_lang = llm.detect_lang(client=client, text=sample)
        assert primary_lang, "Failed to detect primary language"
        tqdm.write(f"Detected language: {primary_lang}")
        cache["lang"] = primary_lang
        sync_cache()

    if primary_lang in SUPPORTED_LANGS:
        total_translations -= total_entries

    processed = 0
    start_time = time.time()
    with tqdm(
        total=total_translations,
        ncols=80,
        dynamic_ncols=True,
        desc="Translating",
        unit="entry",
    ) as pbar:
        for source in source_entries(trans_dir / "main"):
            ns = source.stem
            entries = load_source(source)

            for lang_code, lang_name in SUPPORTED_LANGS.items():
                lang_translations = load_translations(
                    trans_dir=trans_dir,
                    lang=lang_code,
                    ns=ns,
                )
                is_primary_lang = lang_code == primary_lang

                cache_entries = cache.setdefault("entries", {})
                tqdm.write(f"Translating to {lang_name} ({lang_code})...")

                def write():
                    write_translations(
                        trans_dir=trans_dir,
                        lang=lang_code,
                        ns=ns,
                        translations=lang_translations,
                    )
                    sync_cache()

                for key, to_translate, context in entries:

                    cache_entry = cache_entries.setdefault(key, {})
                    cur_input_hash = h(to_translate + (context or ""))

                    langs_hashes = cache_entry.setdefault("hashes", {})
                    lang_hashes = langs_hashes.setdefault(lang_code, {})

                    last_input_hash = lang_hashes.get("input")
                    last_output_hash = lang_hashes.get("output")

                    # If we're the primary language, set us up so that
                    # needs_translation will not be true
                    if is_primary_lang:
                        cur_translation = to_translate
                        last_output_hash = h(to_translate)
                        last_input_hash = cur_input_hash
                    else:
                        cur_translation = lang_translations.get(key)

                    if not cur_translation:
                        needs_translation = True
                    else:
                        in_matches = last_input_hash == cur_input_hash
                        out_matches = last_output_hash == h(cur_translation)
                        hashes_match = in_matches and out_matches
                        needs_translation = not hashes_match

                    if needs_translation:
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
                            tqdm.write(
                                "Translation interrupted by user, quitting")
                            exit()

                        except:  # noqa
                            tqdm.write(
                                f"    Failed to translate `{key}`, using english"
                            )
                            updated_translation = to_translate
                        else:
                            tqdm.write(
                                f'    Translated `{key}` to "{updated_translation}"'
                            )
                    else:
                        updated_translation = cur_translation

                    lang_hashes["output"] = h(updated_translation)
                    lang_hashes["input"] = cur_input_hash
                    lang_translations[key] = updated_translation
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
