#!/usr/bin/env python3
"""
Pull every example conversation out of the OpenAI Model Spec web page and save it as JSON.

Each example in the spec is a conversation followed by one or more reference responses
marked GOOD (compliant), OK (acceptable), or BAD (violation), usually with a short
commentary. Those references plus the surrounding section text are the rubric.

Multi-turn examples (graded responses, then another user turn, then more graded
responses) become one item per graded turn; earlier turns use the first GOOD response
as the assistant's reply, the way the spec reads.

  uv run extract_spec.py                                   # fetches the 2026-10-29 spec page
  uv run extract_spec.py https://model-spec.openai.com/<version>.html
  uv run extract_spec.py saved-page.html                   # or a saved .html / Safari .webarchive

Prefer the published page to a browser-saved copy.
"""
import argparse
import json
import plistlib
import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

DEFAULT_SOURCE = "https://model-spec.openai.com/2026-10-29.html"
SECTION_TEXT_LIMIT = 4000
BLOCKS = {"p", "div", "pre", "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "blockquote", "table", "tr", "br"}


def load_html(path):
    if str(path).startswith(("http://", "https://")):
        import httpx
        response = httpx.get(str(path), follow_redirects=True, timeout=60)
        response.raise_for_status()
        return response.text, str(path)
    data = Path(path).read_bytes()
    if data.startswith(b"bplist"):
        archive = plistlib.loads(data)
        main = archive["WebMainResource"]
        return main["WebResourceData"].decode(main.get("WebResourceTextEncodingName") or "utf-8"), \
            main.get("WebResourceURL", "")
    return data.decode("utf-8"), ""


def to_text(node):
    """HTML to plain text that keeps paragraphs, list bullets, and code blocks."""
    if node is None:
        return ""
    out = []

    def walk(n):
        if isinstance(n, NavigableString):
            out.append(str(n))
            return
        if not isinstance(n, Tag) or n.name in ("svg", "style", "script"):
            return
        if n.name == "pre":
            out.append("\n" + n.get_text() + "\n")
            return
        if n.name == "br":
            out.append("\n")
            return
        if n.name == "li":
            out.append("\n- ")
        elif n.name in BLOCKS:
            out.append("\n")
        for child in n.children:
            walk(child)
        if n.name in BLOCKS:
            out.append("\n")

    walk(node)
    text = "".join(out)
    lines = [re.sub(r"[ \t]+", " ", line).strip() if not line.startswith("    ") else line.rstrip()
             for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def role_of(wrapper):
    """Map a message wrapper to (kind, role, tool_name)."""
    classes = wrapper.get("class", [])
    name = wrapper.select_one(".role-name").get_text(" ", strip=True)
    if "prefix-context" in classes:
        kind = "context"
    else:
        kind = next((k for k in ("good", "ok", "bad") if k in classes), "context")
    tool = None
    if name.startswith("Assistant to "):
        role, tool = "assistant", name[len("Assistant to "):].replace("Function Call ", "").strip()
    else:
        role = name.lower()
    return kind, role, tool


def heading_text(h):
    if h is None:
        return ""
    h = BeautifulSoup(str(h), "html.parser")
    for badge in h.select(".section-badges"):
        badge.decompose()
    return h.get_text(" ", strip=True)


def section_info(conv):
    section = conv.find_parent("div", class_="section")
    heading = section.find(re.compile(r"^h[1-6]$")) if section else None
    authority = [a.get_text(strip=True) for a in heading.select(".authority")] if heading else []
    # Breadcrumb: walk back through earlier headings of higher rank.
    path, rank = [], int(heading.name[1]) if heading else 9
    for h in (heading or conv).find_all_previous(re.compile(r"^h[1-6]$")):
        r = int(h.name[1])
        if r < rank:
            path.insert(0, heading_text(h))
            rank = r
        if rank == 1:
            break
    # Spec prose in this section up to the example (the rubric's "why").
    prose = []
    if section:
        for child in section.children:
            if child is conv or (isinstance(child, Tag) and conv in child.descendants):
                break
            if isinstance(child, Tag) and "conversation" not in child.get("class", []) and child is not heading:
                prose.append(to_text(child))
    text = "\n\n".join(p for p in prose if p)
    if len(text) > SECTION_TEXT_LIMIT:
        text = text[:SECTION_TEXT_LIMIT].rsplit(" ", 1)[0] + " […]"
    return {
        "section_id": section.get("id") if section else "",
        "section": heading_text(heading),
        "path": path,
        "authority": authority[0] if authority else "",
        "section_text": text,
    }


def parse(html):
    soup = BeautifulSoup(html, "html.parser")
    items = []
    for n, conv in enumerate(soup.select("div.conversation")):
        figure = conv.select_one(".figure")
        title = figure.get_text(" ", strip=True) if figure else ""
        u18 = bool(conv.select_one(".u18-badge"))
        info = section_info(conv)

        messages, turn, graded = [], 0, []

        def flush():
            nonlocal turn, graded
            if not graded:
                return
            turn += 1
            refs = [{"quality": k, "content": c, "commentary": com, "tool": t} for k, _, c, com, t in graded]
            items.append({
                "id": f"{n:03d}-{turn}" if turn > 1 or multi_turn else f"{n:03d}",
                "conversation_index": n,
                "turn": turn,
                "title": title,
                "u18": u18,
                **info,
                "messages": [dict(m) for m in messages],
                "references": refs,
            })
            first_good = next((r for r in refs if r["quality"] == "good"), refs[0])
            messages.append({"role": "assistant", "content": first_good["content"],
                             **({"tool": first_good["tool"]} if first_good["tool"] else {})})
            graded = []

        wrappers = conv.select("div.message-wrapper")
        kinds = [role_of(w)[0] for w in wrappers]
        first_graded = next((i for i, k in enumerate(kinds) if k != "context"), None)
        multi_turn = first_graded is not None and "context" in kinds[first_graded:]

        for w in wrappers:
            kind, role, tool = role_of(w)
            content = to_text(w.select_one(".inner-content"))
            commentary = w.select_one(".commentary").get_text(" ", strip=True)
            if kind == "context":
                flush()
                msg = {"role": role, "content": content}
                if tool:
                    msg["tool"] = tool
                messages.append(msg)
            else:
                graded.append((kind, role, content, commentary, tool))
        flush()
    return items


def main():
    parser = argparse.ArgumentParser(description="Extract Model Spec examples to JSON")
    parser.add_argument("source", nargs="?", default=DEFAULT_SOURCE, help="Spec page URL, .html or .webarchive")
    parser.add_argument("-o", "--out", default=str(Path(__file__).resolve().parent / "spec_examples.json"))
    args = parser.parse_args()

    html, url = load_html(args.source)
    items = parse(html)
    Path(args.out).write_text(json.dumps({"source": url or args.source, "items": items},
                                         indent=1, ensure_ascii=False))
    agentic = sum(any(m["role"] in ("tool",) or m.get("tool") or m["content"].startswith("[…")
                      for m in it["messages"] if m["role"] != "user") for it in items)
    print(f"{len(items)} graded items from {len({i['conversation_index'] for i in items})} examples "
          f"({sum(i['u18'] for i in items)} U18, {agentic} agentic/tool) -> {args.out}")


if __name__ == "__main__":
    main()
