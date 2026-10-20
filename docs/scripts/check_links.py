#!/usr/bin/env python3
"""Check rendered local links, resources and fragments without network access."""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Document(HTMLParser):
    def __init__(self, text: str):
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("name"):
            self.ids.add(attrs["name"])
        for key in ("href", "src", "poster"):
            if attrs.get(key):
                self.links.append(attrs[key])


def check(root: Path) -> list[str]:
    root = root.resolve()
    pages = {p.resolve(): Document(p.read_text()) for p in root.rglob("*.html")}
    errors = []
    if not pages or not (root / "index.html").is_file():
        errors.append("No rendered index.html; render the complete book first")
    for source, document in pages.items():
        for link in document.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            path = unquote(url.path)
            if path.startswith("/pmf-glm/"):
                target = root / path.removeprefix("/pmf-glm/")
            elif path.startswith("/"):
                target = root / path.lstrip("/")
            else:
                target = source.parent / path if path else source
            target = target.resolve()
            if target.is_dir():
                target = target / "index.html"
            prefix = f"{source.relative_to(root)}: {link}"
            if not target.is_relative_to(root) or not target.is_file():
                errors.append(prefix + " (missing local target)")
            elif url.fragment and target in pages:
                fragment = unquote(url.fragment).split(":~:text=", 1)[0]
                if fragment and fragment not in pages[target].ids:
                    errors.append(prefix + " (missing fragment)")
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path)
    args = parser.parse_args()
    errors = check(args.site)
    if errors:
        raise SystemExit("\n".join(errors))
    print("All rendered local page, asset and fragment links resolve.")
