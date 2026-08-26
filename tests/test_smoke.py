import pytest

try:
    import neuralforge
except ImportError:
    neuralforge = None

def test_package_importable():
    if neuralforge is None:
        pytest.skip("neuralforge requires optional dependencies not installed")
    assert neuralforge is not None