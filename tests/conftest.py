"""Pytest configuration and fixtures."""

import pytest
from dataclasses import dataclass, field
from typing import List

# Define a compatible TestReport class so we don't have to import from the script
# which might cause side effects or import errors if dependencies differ.
@dataclass
class TestReportFixture:
    """Aggregated metrics from all tests (Fixture version)."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeouts: int = 0
    non_json_responses: int = 0
    latencies_ms: list = field(default_factory=list)
    errors: list = field(default_factory=list)

@pytest.fixture(scope="module")
def report():
    """Fixture providing a TestReport instance for stress tests."""
    return TestReportFixture()
