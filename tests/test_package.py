"""Foundation smoke test. Does not test any recommendation functionality."""

from importlib.metadata import version

import recsys


def test_package_is_importable() -> None:
    assert recsys.__version__ == version("evidence-grounded-recsys")
