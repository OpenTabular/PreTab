"""Executes the python code fences in the tutorial pages so API drift breaks CI
instead of the docs quietly going stale.

Notebook execution (``myst-nb`` / ``nbmake``) is intentionally out of scope for
1.0; this is the lighter-weight alternative: each tutorial's ```python blocks run
in one shared namespace, in source order, exactly as written on the page. Blocks
that only show expected console output (```text fences) are not touched.
"""

import re
from pathlib import Path

import pytest

TUTORIALS_DIR = Path(__file__).parents[2] / "docs" / "tutorials"

_FENCE = re.compile(r"^```python\n(.*?)^```\s*$", re.DOTALL | re.MULTILINE)


def _code_blocks(path: Path) -> list[tuple[int, str]]:
    """Return (1-based start line, source) for every python fence in ``path``.

    The source is padded with leading blank lines so a traceback raised while
    executing it reports the real line number in the markdown file.
    """
    text = path.read_text(encoding="utf-8")
    blocks = []
    for match in _FENCE.finditer(text):
        start_line = text.count("\n", 0, match.start()) + 2
        padded = "\n" * (start_line - 1) + match.group(1)
        blocks.append((start_line, padded))
    return blocks


_TUTORIALS = sorted(TUTORIALS_DIR.glob("*.md"))


@pytest.mark.parametrize("tutorial", _TUTORIALS, ids=[p.stem for p in _TUTORIALS])
def test_tutorial_code_runs(tutorial):
    namespace: dict = {"__name__": "__main__"}
    for start_line, source in _code_blocks(tutorial):
        try:
            exec(compile(source, str(tutorial), "exec"), namespace)  # noqa: S102
        except Exception as exc:
            raise AssertionError(f"{tutorial.name}:{start_line} raised {exc!r}") from exc
