"""
Unit tests for the search_code tool.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.repository.workspace import Workspace
from agent.tools.search_code import search_code


@pytest.fixture
def java_repo(tmp_path: Path) -> Path:
    """Create a minimal Java-like repo."""
    src = tmp_path / "src"
    src.mkdir()

    (src / "OrderService.java").write_text(
        "public class OrderService {\n"
        "    public Order createOrder(Long customerId) {\n"
        "        customerService.validateCustomer(customerId);\n"
        "        return new Order(customerId);\n"
        "    }\n"
        "}\n"
    )
    (src / "CustomerService.java").write_text(
        "public class CustomerService {\n"
        "    public void validateCustomer(Long customerId) {\n"
        "        if (customer.getStatus() == CustomerStatus.ACTIVE) {\n"
        "            throw new CustomerNotEligibleException();\n"
        "        }\n"
        "    }\n"
        "}\n"
    )
    (src / "CustomerStatus.java").write_text(
        "public enum CustomerStatus {\n"
        "    PENDING, ACTIVE, SUSPENDED, CLOSED;\n"
        "}\n"
    )
    # Binary file — should be skipped
    (src / "compiled.class").write_bytes(b"\x00\x01\x02\x03binary content")
    return tmp_path


def test_search_finds_match(java_repo: Path) -> None:
    ws = Workspace(java_repo)
    result = search_code(ws, "createOrder")

    assert "error" not in result
    assert result["total_matches"] >= 1
    files = {m["file"] for m in result["matches"]}
    assert any("OrderService" in f for f in files)


def test_search_returns_line_numbers(java_repo: Path) -> None:
    ws = Workspace(java_repo)
    result = search_code(ws, "validateCustomer")

    assert result["total_matches"] >= 1
    for m in result["matches"]:
        assert isinstance(m["line"], int)
        assert m["line"] >= 1


def test_search_returns_context(java_repo: Path) -> None:
    ws = Workspace(java_repo)
    result = search_code(ws, "CustomerNotEligibleException")

    assert result["total_matches"] >= 1
    m = result["matches"][0]
    assert "context_before" in m
    assert "context_after" in m


def test_search_with_file_pattern(java_repo: Path) -> None:
    ws = Workspace(java_repo)
    result = search_code(ws, "ACTIVE", file_pattern="*.java")

    assert "error" not in result
    # Should only find in Java files
    for m in result["matches"]:
        assert m["file"].endswith(".java")


def test_search_empty_query(java_repo: Path) -> None:
    ws = Workspace(java_repo)
    result = search_code(ws, "")

    assert "error" in result


def test_search_no_matches(java_repo: Path) -> None:
    ws = Workspace(java_repo)
    result = search_code(ws, "XYZZY_NONEXISTENT_TOKEN_99999")

    assert result["total_matches"] == 0
    assert result["matches"] == []


def test_search_skips_binary(java_repo: Path) -> None:
    ws = Workspace(java_repo)
    result = search_code(ws, "binary")

    # binary content should not appear in results
    for m in result["matches"]:
        assert not m["file"].endswith(".class")


def test_search_scalar_and_malformed_inputs(java_repo: Path) -> None:
    ws = Workspace(java_repo)
    # Integer query
    res_int = search_code(ws, 500)
    assert "error" not in res_int
    assert res_int["query"] == "500"

    # Float query
    res_float = search_code(ws, 12.34)
    assert "error" not in res_float
    assert res_float["query"] == "12.34"

    # Invalid regex query
    res_bad_regex = search_code(ws, "[unclosed-bracket", is_regex=True)
    assert "error" in res_bad_regex
    assert "regular expression" in res_bad_regex["error"].lower() or "regex" in res_bad_regex["error"].lower()

    # None or invalid type
    res_none = search_code(ws, None)
    assert "error" in res_none

    # Malformed file pattern does not crash
    res_pattern = search_code(ws, "ACTIVE", file_pattern="[a-")
    assert isinstance(res_pattern, dict)

