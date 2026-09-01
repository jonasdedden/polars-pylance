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
    if parts[-1] == "__init__":
        parts = parts[:-1]
    # Underscore-prefixed modules are the implementation, not the API.
    if not parts or any(p.startswith("_") for p in parts):
        continue
    doc = Path(*parts).with_suffix(".md")
    nav[parts] = doc.as_posix()
    with mkdocs_gen_files.open(Path("reference", doc), "w") as f:
        f.write(f"::: {'.'.join(parts)}\n")

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as f:
    f.writelines(nav.build_literate_nav())
