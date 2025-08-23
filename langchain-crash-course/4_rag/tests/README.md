# To run the tests

## 1. Create a test directory structure

```
langchain-crash-course/
├── 4_rag/
│ ├── tests/
│ │ ├── **init**.py
│ │ ├── conftest.py
│ │ ├── test_rag_with_metadata.py
│ │ ├── test_books/ (created by tests)
│ │ └── test_db/ (created by tests)
│ ├── utils/
│ │ ├── **init**.py
│ │ └── logger.py
│ └── rag_with_metadata.py
```

## 2. Run the tests using pytest

```pwsh
cd langchain-crash-course/4_rag 
pytest tests/ -v
```

## The tests will

- Verify document loading functionality
- Test text chunking
- Check vector store initialization behavior when store already exists
- Validate the main application flow
- Ensure proper logging throughout the process

## The test suite

- Creates isolated test directories
- Cleans up after itself
- Verifies logging behavior
- Checks core functionality
- Uses fixtures for setup/teardown
- Follows testing best practices

## Expected log output in rag_application.log will include

- INFO level messages for normal operations
- WARNING level messages for non-critical issues
- ERROR level messages for any test error conditions
- The tests assume the vector store already exists and verify that the code handles this case correctly while properly logging all operations.
