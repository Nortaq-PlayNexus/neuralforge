import importlib
import pkgutil
import pytest

import neuralforge


def test_package_importable():
    assert neuralforge is not None


def test_import_all_submodules():
    errors = []
    for mod in pkgutil.walk_packages(neuralforge.__path__, neuralforge.__name__ + "."):
        try:
            importlib.import_module(mod.name)
        except Exception as exc:
            errors.append(f"{mod.name}: {exc}")
    if errors:
        pytest.fail("Submodule import failures:\n" + "\n".join(errors))
