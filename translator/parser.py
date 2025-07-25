import re

from lxml import etree

_parser = etree.XMLParser(
    resolve_entities=False,
    load_dtd=False,
    recover=True,
    encoding="utf-8",
)


def render_inner(node: etree._Element) -> str:
    """Render the inner text and markup of an XML node, minus the outer tags."""
    result = []

    # Some text may come before the first child
    if node.text:
        result.append(node.text)

    # Add each child's string representation
    for child in node:
        result.append(etree.tostring(child, encoding="unicode"))

    return "".join(result)


def escape_entities(xml_str: str):
    """Replace unescaped ampersands with &amp; in an XML string. Lots of
    responses have amperands that are not escaped."""
    # Pattern to find & that are not already part of an entity. Negative
    # lookbehind & lookahead assertions are used.
    pattern = re.compile(r"&(?!amp;|lt;|gt;|quot;|#)")
    # Replace those ampersands with &amp;
    return pattern.sub(r"&amp;", xml_str)


def parse_document(s: str) -> etree._Element:
    escaped = escape_entities(s)
    root = etree.fromstring(escaped, _parser)
    return root


def find_and_parse_xml(s: str, root_tag="root") -> etree._Element:
    """For a given string, find the first XML document and parse it."""
    start_tag = f"<{root_tag}>"
    end_tag = f"</{root_tag}>"

    start_index = s.find(start_tag)
    end_index = s.find(end_tag)

    if start_index == -1 or end_index == -1 or start_index > end_index:
        raise ValueError("No valid XML document found: " + s)

    xml_document = s[start_index : end_index + len(end_tag)]
    try:
        root = parse_document(xml_document)

    # This basically never happens now that we have recover=True on the parser
    except etree.XMLSyntaxError as e:
        raise ValueError("Invalid XML document: " + str(e))

    return root


def escape_placeholder_tags(xml):
    # Replace opening tags
    xml = re.sub(r"<(\d+)>", r"<tag_\1>", xml)
    # Replace closing tags
    xml = re.sub(r"</(\d+)>", r"</tag_\1>", xml)
    return xml


def unescape_placeholder_tags(xml):
    # Replace opening tags
    xml = re.sub(r"<tag_(\d+)>", r"<\1>", xml)
    # Replace closing tags
    xml = re.sub(r"</tag_(\d+)>", r"</\1>", xml)
    return xml
