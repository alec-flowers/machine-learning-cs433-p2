import pathlib

REPO_ROOT = pathlib.Path(__file__).absolute().parents[1].absolute().resolve()
DATA_DIR = (pathlib.Path(__file__).absolute(
).parents[1] / "Data").absolute().resolve()
assert(REPO_ROOT.exists())
assert(DATA_DIR.exists())