# test_agent_tools_basic.py
"""
Unit tests for the agent_tools_basic.py module.

This test suite verifies the core functionalities of the basic agent
application, including tool functionality, configuration handling, and
agent component setup.
"""

# Import standard libraries
import importlib
import os
import re
import sys
from pathlib import Path

# Import pytest
import pytest


# Helper to import the module with a clean environment
def import_module_with_env(env_vars):
    """Import agent_tools_basic.py after setting env variables."""
    # Backup current env
    backup_env = {k: os.getenv(k) for k in env_vars}

    # Add parent directory to sys.path to allow for module import
    module_path = str(Path(__file__).parent.parent.resolve())
    sys.path.insert(0, module_path)

    try:
        # Explicitly clear any existing LLM env vars
        for var in ("OPENAI_API_KEY", "OPENAI_LLM", "OLLAMA_LLM"):
            os.environ.pop(var, None)

        # Set the env vars supplied by the test
        for k, v in env_vars.items():
            os.environ[k] = v

        # Invalidate any cached import
        if "agent_tools_basic" in sys.modules:
            del sys.modules["agent_tools_basic"]
        module = importlib.import_module("agent_tools_basic")
        return module
    finally:
        # Restore original env
        for k, v in backup_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        # Clean up sys.path
        if sys.path and sys.path[0] == module_path:
            sys.path.pop(0)


@pytest.fixture
def agent_module(monkeypatch):
    """Fixture that imports the module with dummy LLM configuration."""
    # Provide dummy values for OpenAI or Ollama
    monkeypatch.setenv("OPENAI_API_KEY", "dummy_key")
    monkeypatch.setenv("OPENAI_LLM", "dummy-model")
    # Ensure no Ollama config interferes
    monkeypatch.delenv("OLLAMA_LLM", raising=False)
    return import_module_with_env(
        {"OPENAI_API_KEY": "dummy_key", "OPENAI_LLM": "dummy-model"}
    )


def test_get_current_time_format(agent_module):
    """get_current_time should return a string in YYYY-MM-DD HH:MM:SS format."""
    current_time = agent_module.get_current_time()
    assert isinstance(current_time, str)
    # Regex for the expected datetime format
    pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
    assert re.match(pattern, current_time), f"Unexpected format: {current_time}"


def test_tools_list_contains_current_time(agent_module):
    """The tools list must contain a Tool named 'Current Time'."""
    tools = agent_module.tools
    assert isinstance(tools, list)
    # Find tool by name
    names = [tool.name for tool in tools]
    assert "Time" in names, f"Tool 'Time' not found in {names}"


def test_module_import_with_missing_llm(monkeypatch):
    """Importing the module without any LLM configuration should raise ValueError."""
    # Remove all relevant env vars
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_LLM", raising=False)
    monkeypatch.delenv("OLLAMA_LLM", raising=False)

    # Patch dotenv.load_dotenv to prevent loading from a .env file,
    # which would interfere with this test.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)

    with pytest.raises(ValueError) as excinfo:
        import_module_with_env({})
    assert "Neither OpenAI" in str(excinfo.value)
