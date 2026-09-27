import sys
from pathlib import Path

# make `import config` and `import pipeline` work when running pytest from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib
import pkgutil

import pytest


@pytest.fixture(autouse=True)
def _review_files_to_tmp(tmp_path, monkeypatch):
    """
    Steps write review csv files (data/review/...) as a side effect, e.g. marketcap.build() ->
    corporate_actions_review.csv. Redirect every such path of the pipeline modules to tmp_path so
    tests never overwrite the real review files.
    """
    import config
    import pipeline
    review = Path(config.REVIEW_DIR).resolve()
    for m in pkgutil.iter_modules(pipeline.__path__):
        mod = importlib.import_module(f'pipeline.{m.name}')
        for name, value in list(vars(mod).items()):
            if isinstance(value, Path) and value.suffix == '.csv' and value.resolve().parent == review:
                monkeypatch.setattr(mod, name, tmp_path / value.name)
