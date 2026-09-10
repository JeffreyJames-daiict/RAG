"""Safe mathematical calculator tool for agentic workflows."""

from __future__ import annotations

import ast
import operator
from typing import Any, Dict
from pydantic import BaseModel, Field

from rag.agentic.tools.base import BaseTool

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an AST expression using only allowed math operators."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _OPERATORS[type(node.op)](left, right)
    raise ValueError(f"Unsupported math syntax in expression: {ast.dump(node)}")


class CalculatorInput(BaseModel):
    """Input expression for calculator."""

    expression: str = Field(..., description="Mathematical expression (e.g., '1500 * 1.15 / 12')")


class CalculatorOutput(BaseModel):
    """Result of mathematical computation."""

    expression: str = Field(..., description="Original input expression")
    result: float = Field(..., description="Computed numerical result")
    formatted: str = Field(..., description="Human-readable formatted result")


class CalculatorTool(BaseTool):
    """Safe mathematical calculator for calculating metrics, percentages, and aggregates."""

    name = "calculator"
    description = "Computes mathematical expressions safely (addition, subtraction, multiplication, division, powers)."
    input_schema = CalculatorInput
    output_schema = CalculatorOutput

    def execute(self, inputs: CalculatorInput) -> CalculatorOutput:
        """Parse and evaluate math expression safely."""
        expr = inputs.expression.strip().replace(",", "")
        try:
            tree = ast.parse(expr, mode="eval")
            res = _safe_eval(tree)
            return CalculatorOutput(
                expression=inputs.expression,
                result=res,
                formatted=f"{res:,.4f}".rstrip("0").rstrip("."),
            )
        except Exception as exc:
            raise ValueError(f"Calculator failed to evaluate '{inputs.expression}': {exc}")
