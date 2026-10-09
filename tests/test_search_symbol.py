"""
Unit tests for the search_symbol tool.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.repository.workspace import Workspace
from agent.tools.search_symbol import search_symbol


@pytest.fixture
def symbol_repo(tmp_path: Path) -> Path:
    """Create a repo with Java and Python source files."""
    src = tmp_path / "src"
    src.mkdir()

    (src / "OrderService.java").write_text(
        "package com.example;\n"
        "\n"
        "public class OrderService {\n"
        "\n"
        "    public Order createOrder(Long customerId) {\n"
        "        return new Order();\n"
        "    }\n"
        "\n"
        "    public void cancelOrder(Long orderId) {\n"
        "        // cancel\n"
        "    }\n"
        "}\n"
    )
    (src / "CustomerService.java").write_text(
        "public class CustomerService {\n"
        "    public void validateCustomer(Long id) {}\n"
        "}\n"
    )
    (src / "utils.py").write_text(
        "class OrderProcessor:\n"
        "    def process_order(self, order_id):\n"
        "        pass\n"
        "\n"
        "def validate_customer(customer_id):\n"
        "    pass\n"
    )
    return tmp_path


def test_find_java_class(symbol_repo: Path) -> None:
    ws = Workspace(symbol_repo)
    result = search_symbol(ws, "OrderService")

    assert "error" not in result
    assert result["total_results"] >= 1
    names = [r["symbol"] for r in result["results"]]
    assert "OrderService" in names


def test_find_java_method(symbol_repo: Path) -> None:
    ws = Workspace(symbol_repo)
    result = search_symbol(ws, "createOrder")

    assert "error" not in result
    assert result["total_results"] >= 1
    r = result["results"][0]
    assert r["start_line"] >= 1
    assert r["end_line"] >= r["start_line"]


def test_find_python_class(symbol_repo: Path) -> None:
    ws = Workspace(symbol_repo)
    result = search_symbol(ws, "OrderProcessor")

    assert "error" not in result
    assert result["total_results"] >= 1
    r = result["results"][0]
    assert r["symbol_type"] == "class"


def test_find_python_function(symbol_repo: Path) -> None:
    ws = Workspace(symbol_repo)
    result = search_symbol(ws, "validate_customer")

    assert "error" not in result
    assert result["total_results"] >= 1


def test_symbol_not_found(symbol_repo: Path) -> None:
    ws = Workspace(symbol_repo)
    result = search_symbol(ws, "NonExistentSymbol999")

    assert "error" not in result
    assert result["total_results"] == 0


def test_empty_symbol(symbol_repo: Path) -> None:
    ws = Workspace(symbol_repo)
    result = search_symbol(ws, "")

    assert "error" in result


def test_symbol_with_file_pattern(symbol_repo: Path) -> None:
    ws = Workspace(symbol_repo)
    result = search_symbol(ws, "validate", file_pattern="*.java")

    # Should only find Java matches
    for r in result["results"]:
        assert r["file"].endswith(".java")
