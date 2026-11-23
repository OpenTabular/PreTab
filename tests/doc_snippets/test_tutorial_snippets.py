"""Executes the Python code fences in tutorials and selected documentation pages.

Each page's ```python blocks run in one shared namespace, in source order,
exactly as written on the page. Blocks that only show expected console output
(```text fences) are not touched. The companion notebook is kept in sync with its
executed tutorial section without notebook-rendering dependencies in CI.
"""

import json
import re
from importlib.util import find_spec
from pathlib import Path

import pytest

from pretab.compose import registry

ROOT = Path(__file__).parents[2]
TUTORIALS_DIR = ROOT / "docs" / "tutorials"
CORE_CONCEPTS_DIR = ROOT / "docs" / "core_concepts"
REPRESENTATIONS_DIR = ROOT / "docs" / "representations"

_FENCE = re.compile(r"^```python\n(.*?)^```\s*$", re.DOTALL | re.MULTILINE)


def _code_blocks(path: Path, section: str | None = None) -> list[tuple[int, str]]:
    """Return (1-based start line, source) for every python fence in ``path``.

    Optionally select one second-level section. The source is padded with leading
    blank lines so a traceback reports the real line number in the markdown file.
    """
    text = path.read_text(encoding="utf-8")
    start = text.index(f"## {section}\n") if section is not None else 0
    stop = text.find("\n## ", start + 1) if section is not None else -1
    stop = len(text) if stop == -1 else stop
    blocks = []
    for match in _FENCE.finditer(text):
        if not start <= match.start() < stop:
            continue
        start_line = text.count("\n", 0, match.start()) + 2
        padded = "\n" * (start_line - 1) + match.group(1)
        blocks.append((start_line, padded))
    return blocks


_PAGES = [
    *sorted(TUTORIALS_DIR.glob("*.md")),
    *sorted(CORE_CONCEPTS_DIR.glob("*.md")),
    REPRESENTATIONS_DIR / "functional_expansions.md",
    REPRESENTATIONS_DIR / "kernel_approximation.md",
    REPRESENTATIONS_DIR / "numerical_encoding.md",
    REPRESENTATIONS_DIR / "spline_expansions.md",
    ROOT / "README.md",
    ROOT / "docs" / "homepage.md",
    ROOT / "docs" / "getting_started" / "overview.md",
    ROOT / "docs" / "getting_started" / "quickstart.md",
    ROOT / "docs" / "developer_guide" / "architecture.md",
]


@pytest.fixture(autouse=True)
def isolate_document_state(tmp_path, monkeypatch):
    """Keep example files and representation registrations local to each page."""
    monkeypatch.chdir(tmp_path)
    transformers = registry.TRANSFORMER_REGISTRY.copy()
    numerical = registry.NUMERICAL_METHODS.copy()
    categorical = registry.CATEGORICAL_METHODS.copy()
    yield
    registry.TRANSFORMER_REGISTRY.clear()
    registry.TRANSFORMER_REGISTRY.update(transformers)
    registry.NUMERICAL_METHODS.clear()
    registry.NUMERICAL_METHODS.update(numerical)
    registry.CATEGORICAL_METHODS.clear()
    registry.CATEGORICAL_METHODS.update(categorical)


@pytest.mark.parametrize(
    ("tutorial", "section"),
    [pytest.param(page, None, id=page.stem) for page in _PAGES]
    + [
        pytest.param(
            TUTORIALS_DIR / "multivariate_features.md",
            "Grouped preprocessing with Preprocessor",
            id="feature-groups",
        ),
    ],
)
@pytest.mark.filterwarnings("ignore::pretab.exceptions.LeakageWarning")
def test_tutorial_code_runs(tutorial, section):
    namespace: dict = {"__name__": "__main__"}
    blocks = _code_blocks(tutorial, section)
    if tutorial == CORE_CONCEPTS_DIR / "configuration.md" and find_spec("polars") is None:
        polars_lines = {line for line, _ in _code_blocks(tutorial, "Polars DataFrame input")}
        blocks = [(line, source) for line, source in blocks if line not in polars_lines]
    assert blocks, f"No Python examples found in {tutorial.name}: {section}"
    for start_line, source in blocks:
        try:
            exec(compile(source, str(tutorial), "exec"), namespace)  # noqa: S102
        except Exception as exc:
            raise AssertionError(f"{tutorial.name}:{start_line} raised {exc!r}") from exc


def test_per_feature_notebook_matches_tutorial():
    notebook_path = ROOT / "examples" / "per-feature-configuration.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook_sources = ["".join(cell["source"]).strip() for cell in notebook["cells"] if cell["cell_type"] == "code"]
    tutorial_sources = [
        source.strip()
        for _, source in _code_blocks(
            TUTORIALS_DIR / "sklearn_pipeline.md", "Configure mixed columns with Preprocessor"
        )
    ]
    assert notebook["nbformat"] == 4
    assert len(notebook_sources) == 5
    assert notebook_sources == tutorial_sources
