import pytest
import sys

def test_vertexai_import():
    """Verifies that vertexai (part of google-cloud-aiplatform) can be imported."""
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
    except ImportError as e:
        pytest.fail(f"Failed to import vertexai: {e}")

def test_python_version():
    """Verifies that we are running on a compatible Python version (3.7+)."""
    assert sys.version_info >= (3, 7)
