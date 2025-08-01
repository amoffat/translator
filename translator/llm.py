from dataclasses import dataclass

import openai

from . import parser
from .langs import SUPPORTED_LANGS


@dataclass
class Client:
    openai_client: openai.OpenAI
    model: str = "gpt-4o"


def complete(
    *,
    client: Client,
    prompt: str,
):
    response = client.openai_client.chat.completions.create(
        model=client.model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    resp = response.choices[0].message.content or ""
    return resp


def translate(
    *,
    client: Client,
    dest_code: str,
    to_translate: str,
    context: str | None = None,
):
    dest_lang = SUPPORTED_LANGS[dest_code]

    prompt = f"""
You are a language translation model. You must follow these rules when translating:

* Try to preserve the original meaning and intent of the text.
* Use modern, natural language that is conventional and well-understood in {dest_lang}.
* Do not add new emojis, unless they exist in the text to be translated.
* Preserve all numeric tags, like <1>, <2>, etc. These are i18next html placeholders.
* Do not translate text inside of double curly brace templates: {{{{...}}}}. These are i18next template placeholders and we need to leave them as-is.
* To not translate urls. Leave them as-is.
* Do not translate the <context> tag. Only translate the contents of the <to_translate> tag.
* Use transliteration for names that have no direct translation.
* Use the <context> tag to help you understand the context of the text to be translated.
* Do not directly reference the contents of the <context> tag in the translation. It is only there to help you translate <to_translate> more accurately.
* Always ensure that the contents of the <to_translate> tag is fully translated into {dest_lang}.

Format your response as the following XML document:

<root>
    <translated>
        Translated text goes here, preserving the original meaning and intent.
    </translated>
</root>

Where the contents of <translated> is the translated text.

The text to translate is below in the <to_translate>...</to_translate> tags.
Please translate it to {dest_lang}.

<translate>
    <context>{context or ''}</context>
    <to_translate>{to_translate}</to_translate>
</translate>
""".strip()

    resp = complete(client=client, prompt=prompt)
    resp = parser.escape_placeholder_tags(resp)
    doc = parser.find_and_parse_xml(resp, root_tag="root")

    text = doc.find("translated")
    if text is not None:
        translated = parser.render_inner(text)
        translated = parser.unescape_placeholder_tags(translated)
        return translated.strip()
    return None


def detect_lang(*, client: Client, text: str) -> str | None:
    prompt = f"""
You are a language detection model. Given the text below, determine the ISO 639
language code (like 'en', 'fr', 'es', etc.) of the text:

<detect>
{text}
</detect>

Format your response as the following XML document:
<detected>
    <lang>...</lang>
</detected>
""".strip()

    resp = complete(client=client, prompt=prompt)
    doc = parser.find_and_parse_xml(resp, root_tag="detected")

    lang = doc.find("lang")
    if lang is not None:
        detected = parser.render_inner(lang)
        detected = parser.unescape_placeholder_tags(detected)
        return detected.strip().lower()
    return None
