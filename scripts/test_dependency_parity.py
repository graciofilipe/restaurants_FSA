"""One place declares the dependencies; `requirements.txt` is generated from it.

The image Cloud Build deploys installs `requirements.txt`. The `.venv` this
repo is developed and tested in comes from `pyproject.toml` and `uv.lock`.
Nothing reconciled the two, and they had drifted in both directions:
`requirements.txt` carried `google-cloud-aiplatform`, which no module imports,
and omitted `fastapi`, `uvicorn` and `pydantic`, which are imported. The lock
was resolved for `>=3.13` — the local interpreter — while every deployed and
CI-tested line runs 3.11 (D12).

These are string checks, not a resolver. They catch the drift that actually
happens: a dependency added to `pyproject.toml` and never exported, or the
runtime version changed in one of the three files that name it. Verifying
resolved versions is `uv lock`'s job, and this file deliberately does not
attempt it — `uv` is not installed in the build image.

Lives under `scripts/` because `pytest app/ scripts/` is the Cloud Build gate,
and a dependency check that the build does not run is a check that fails after
the build breaks.
"""
import pathlib
import re
import tomllib
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent

# The line `uv export` writes at the top of what it generates. Its absence
# means someone edited the file by hand, which is the state D12 ended.
GENERATED_MARKER = 'uv export'


def normalize(name):
    """PEP 503: `Google_Cloud.BigQuery` and `google-cloud-bigquery` are one."""
    return re.sub(r'[-_.]+', '-', name).strip().lower()


def declared_dependencies():
    """The names in `[project].dependencies`, without extras or specifiers."""
    with open(REPO / 'pyproject.toml', 'rb') as f:
        project = tomllib.load(f)['project']
    return {normalize(re.split(r'[\[<>=!~;\s]', spec, maxsplit=1)[0])
            for spec in project['dependencies']}


def exported_requirements():
    """The names pinned in `requirements.txt`, comments and options dropped."""
    names = set()
    for line in (REPO / 'requirements.txt').read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(('#', '-')):
            continue
        names.add(normalize(re.split(r'[\[<>=!~;\s]', line, maxsplit=1)[0]))
    return names


def lower_bound(requires_python):
    """`">=3.11"` -> `(3, 11)`. None if there is no lower bound to read."""
    match = re.search(r'>=\s*(\d+)\.(\d+)', requires_python)
    return (int(match.group(1)), int(match.group(2))) if match else None


class TestRequirementsIsGenerated(unittest.TestCase):

    def test_the_file_says_where_it_came_from(self):
        header = (REPO / 'requirements.txt').read_text()[:600]
        self.assertIn(GENERATED_MARKER, header,
                      'requirements.txt looks hand-written; regenerate it with '
                      '`uv export --no-dev --no-hashes --no-emit-project -o requirements.txt`')

    def test_every_declared_dependency_reaches_the_image(self):
        missing = declared_dependencies() - exported_requirements()
        self.assertEqual(missing, set(),
                         f'declared in pyproject.toml but absent from the image: {sorted(missing)}')


class TestOnePythonVersion(unittest.TestCase):
    """3.11 is named in three files. A mismatch is invisible until a build."""

    def _dockerfile_python(self):
        match = re.search(r'FROM python:(\d+)\.(\d+)',
                          (REPO / 'Dockerfile').read_text())
        return (int(match.group(1)), int(match.group(2)))

    def test_pyproject_declares_the_version_the_image_runs(self):
        with open(REPO / 'pyproject.toml', 'rb') as f:
            requires = tomllib.load(f)['project'].get('requires-python')
        self.assertIsNotNone(
            requires, 'pyproject.toml declares no requires-python, so uv locks '
                      'against whatever interpreter happens to be local')
        self.assertEqual(lower_bound(requires), self._dockerfile_python())

    def test_the_lockfile_was_resolved_for_it(self):
        """A lock resolved for 3.13 can pin a wheel that will not install on
        3.11, and the export inherits it."""
        match = re.search(r'^requires-python = "(.+)"$',
                          (REPO / 'uv.lock').read_text(), re.MULTILINE)
        self.assertIsNotNone(match, 'uv.lock declares no requires-python')
        self.assertEqual(lower_bound(match.group(1)), self._dockerfile_python())

    def test_cloud_build_tests_on_it(self):
        image = re.search(r"name: 'python:(\d+)\.(\d+)",
                          (REPO / 'cloudbuild.yaml').read_text())
        self.assertEqual((int(image.group(1)), int(image.group(2))),
                         self._dockerfile_python())


if __name__ == '__main__':
    unittest.main()
