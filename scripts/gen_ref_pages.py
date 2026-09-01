"""Generate the docs pages that are not written by hand.

Three Markdown files live outside `docs/` because they are read outside the
site too: `README.md` on the repository front page and on PyPI, `COMPARISON.md`
and `bench/README.md` in the repository. They are copied in here, with the
links that only make sense in one of those places rewritten for the other.

There is also one API page per public module, so a new module appears in the
nav without this file or `mkdocs.yml` being touched.
"""

import re
from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
GITHUB = "https://github.com/jonasdedden/polars-pylance/blob/main"

# A link into the published site, with whatever version segment it carries.
# Inside the site the same link has to be relative, or a page served under
# `stable/` would send the reader to `dev/`.
SITE_PAGE = re.compile(r"https://jonasdedden\.github\.io/polars-pylance/[^/]+/(\w+)/")


def publish(text: str, dest: str, rewrites: dict[str, str] | None = None) -> None:
    """Write `text` into the docs tree, applying literal `rewrites` first."""
    for old, new in (rewrites or {}).items():
        text = text.replace(old, new)
    with mkdocs_gen_files.open(dest, "w") as f:
        f.write(text)


def copy_assets(names: list[str], source: Path, dest: str) -> None:
    """Copy plots into the site, since they live outside `docs_dir`."""
    for name in names:
        with mkdocs_gen_files.open(f"{dest}/{name}", "w") as f:
            f.write((source / name).read_text())


# The README keeps absolute links to the published site, because it is rendered
# on GitHub and on PyPI where a relative one would 404. Inside the site those
# same links become relative, so they stay within the version being read.
publish(SITE_PAGE.sub(r"\1.md", (ROOT / "README.md").read_text()), "index.md")

# The comparison's plots are under `bench/`, outside `docs_dir`. Which ones to
# copy is read back out of the page, so a new plot needs no change here.
comparison = (ROOT / "COMPARISON.md").read_text()
publish(
    comparison,
    "COMPARISON.md",
    {
        "bench/plots/static/": "assets/comparison/",
        "](bench/README.md)": "](BENCHMARKS.md)",
    },
)
copy_assets(
    sorted(set(re.findall(r"bench/plots/static/([\w-]+\.svg)", comparison))),
    ROOT / "bench" / "plots" / "static",
    "assets/comparison",
)

# How the benchmarks are run, for anyone reproducing them.
publish(
    (ROOT / "bench" / "README.md").read_text(),
    "BENCHMARKS.md",
    {"](../COMPARISON.md)": "](COMPARISON.md)"},
)

# `PUSHDOWN.md` is a real file in `docs/`, but its plots are not.
copy_assets(
    ["pushdown-none.svg", "pushdown-indexed.svg"],
    ROOT / "bench" / "plots" / "static",
    "assets/bench",
)

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
    # Without an explicit title a page is named after its file, which would
    # render a package's index page as "Index".
    publish(f"---\ntitle: {module}\n---\n\n::: {module}\n", str(Path("reference", doc)))

# `literate-nav` reads this to build the API nav, but it is still rendered as a
# page. Nothing links to it; this keeps it out of the search index too.
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as f:
    f.write("---\nsearch:\n  exclude: true\n---\n\n")
    f.writelines(nav.build_literate_nav())
