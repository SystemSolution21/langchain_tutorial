"""
pytest configuration file for RAG tests.
"""

import pytest
import logging

def pytest_configure(config):
    """
    Configure pytest environment.
    """
    # Ensure logging is properly configured for tests
    logging.basicConfig(level=logging.INFO)

@pytest.fixture(autouse=True)
def log_test_name(request):
    """
    Log the name of each test as it starts.
    """
    logging.info(f"Starting test: {request.node.name}")
    yield
    logging.info(f"Completed test: {request.node.name}")