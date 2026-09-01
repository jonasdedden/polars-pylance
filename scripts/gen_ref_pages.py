"""Generate the docs pages that are not written by hand.

`index.md` is the README, so the landing page cannot drift from it, and there is
one API page per public module, so a new module appears in the nav without this
file or `mkdocs.yml` being touched.
"""

import re
from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"

GITHUB = "https://github.com/jonasdedden/polars-pylance/blob/main"

# The README keeps absolute GitHub links, because it is also rendered on the
# repository front page and on PyPI, where a relative one would 404. Inside the
# site, the comparison is a page of its own, so that one link is pointed at it.
readme = (ROOT / "README.md").read_text()
readme = readme.replace(f"{GITHUB}/COMPARISON.md", "COMPARISON.md")
with mkdocs_gen_files.open("index.md", "w") as f:
    f.write(readme)

# The comparison's plots live under `bench/`, which is not part of `docs_dir`,
# so they are copied in beside the page and the paths rewritten. Which files to
# copy is read back out of the page, so a new plot needs no change here.
comparison = (ROOT / "COMPARISON.md").read_text()
plots = set(re.findall(r"bench/plots/static/([\w-]+\.svg)", comparison))
comparison = comparison.replace("bench/plots/static/", "assets/comparison/")
# `bench/README.md` is not a page here either.
comparison = comparison.replace("](bench/README.md)", f"]({GITHUB}/bench/README.md)")
with mkdocs_gen_files.open("COMPARISON.md", "w") as f:
    f.write(comparison)

for name in sorted(plots):
    svg = ROOT / "bench" / "plots" / "static" / name
    with mkdocs_gen_files.open(f"assets/comparison/{name}", "w") as f:
        f.write(svg.read_text())

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
