"""Generate the docs pages that are not written by hand.

`index.md` is the README, so the landing page cannot drift from it, and there is
one API page per public module, so a new module appears in the nav without this
file or `mkdocs.yml` being touched.
"""

from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"

with mkdocs_gen_files.open("index.md", "w") as f:
    f.write((ROOT / "README.md").read_text())

nav = mkdocs_gen_files.Nav()
for path in sorted(SRC.rglob("*.py")):
    parts = tuple(path.relative_to(SRC).with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts = parts[:-1]
    # Underscore-prefixed modules are the implementation, not the API.
    if not parts or any(p.startswith("_") for p in parts):
        continue
    # A package's own page is the index of its section, so the nav shows one
    # `polars_pylance` entry that opens it rather than a section holding a
    # child of the same name. Relies on the theme's `navigation.indexes`.
    doc = Path(*parts, "index.md") if is_package else Path(*parts).with_suffix(".md")
    nav[parts] = doc.as_posix()
    module = ".".join(parts)
    with mkdocs_gen_files.open(Path("reference", doc), "w") as f:
        # Without an explicit title a page is named after its file, which would
        # render a package's index page as "Index".
        f.write(f"---\ntitle: {module}\n---\n\n::: {module}\n")

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as f:
    # `literate-nav` reads this to build the API nav, but it is still rendered
    # as a page. Nothing links to it; this keeps it out of the search index too.
    f.write("---\nsearch:\n  exclude: true\n---\n\n")
    f.writelines(nav.build_literate_nav())
