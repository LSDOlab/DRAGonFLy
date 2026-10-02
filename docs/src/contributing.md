# Contributing

## Pull requests
Work on a branch and open a pull request into `main`. Before opening it:
1. Merge the latest `main` into your branch and resolve conflicts.
2. Run the tests: `OMP_NUM_THREADS=1 pytest` (add `-rP` to show print output).
3. Add tests for new features, and comment the code.
4. Fill out the pull request template.

A reviewer approves when the tests pass on GitHub Actions and locally, and the changes are clear and commented.
When asking for changes, describe precisely what needs to be fixed.

## Tests
Tests live in `tests/`: files named `test_*.py` with functions named `test_*`, run by pytest. They must work on
any number of MPI ranks: every collective is called by every rank, and assertions are reduced over the entities
each rank owns.

## Documentation
The website is built with Sphinx from the Markdown (MyST) pages in `docs/src/` and published to
<https://lsdolab.github.io/DRAGonFLy> by the `docs` GitHub Actions workflow on every push to `main`.

- Add a page as a `.md` file in `docs/src/` and list it in the toctree of `docs/src/welcome.md`.
- Equations use `$...$` and `$$...$$`; references go in `docs/src/references.bib` and are cited with
  ``{cite:p}`key` ``.
- The API reference is generated from the docstrings in `dragonfly_sim`.
- Build locally with `pip install -r requirements.txt`, then `make html` in `docs/`, and open
  `docs/_build/html/index.html`.

## Releases
Update `__version__` in `dragonfly_sim/__init__.py` and add an entry to [Release notes](release_notes.md).
