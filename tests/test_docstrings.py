"""What the docstrings have to look like for `mkdocstrings` to render them.

The docstrings are Markdown, so constructs that are inert in a plain text
docstring are not inert here. `properdocs build --strict` does not catch these:
a doctest that Markdown reads as a blockquote is wrong, not broken, so the
build succeeds and only the rendered page shows it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import polars_pylance

SRC = Path(polars_pylance.__file__).parent


def _docstrings() -> list[tuple[str, str]]:
    """Every docstring in the package, as `(where, text)`.

    Globbed rather than walked with `pkgutil`, which skips `__init__.py`: the
    package docstring is the front page of the site and the one this first
    went wrong in.
    """
    found: list[tuple[str, str]] = []
    for path in sorted(SRC.glob("*.py")):
        tree = ast.parse(path.read_text())
        nodes: list[tuple[str, ast.AST]] = [("module", tree)]
        nodes += [
            (getattr(n, "name", "?"), n)
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        for name, node in nodes:
            doc = ast.get_docstring(node)  # type: ignore[arg-type]
            if doc:
                found.append((f"{path.name}:{name}", doc))
    return found


@pytest.mark.parametrize(("where", "doc"), _docstrings(), ids=lambda v: v[:60])
def test_doctests_are_not_markdown_blockquotes(where: str, doc: str) -> None:
    """A `>>>` at column 0 is three nested blockquotes, not a code block.

    Indenting it under `Examples:` is enough, and so is a ```pycon fence; both
    keep the prompts, so `--doctest-modules` still runs the example either way.
    """
    offenders: list[str] = []
    fenced = False
    for line in doc.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        # Indented lines are already a code block; fenced ones are explicit.
        if not fenced and line.startswith((">>>", "... ")):
            offenders.append(line)
    assert not offenders, (
        f"{where}: doctest lines at column 0 render as a blockquote. "
        f"Indent them under `Examples:` or fence them with ```pycon. "
        f"First offender: {offenders[0]!r}"
    )
