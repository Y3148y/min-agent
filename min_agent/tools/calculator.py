"""``calculator`` -- arithmetic via a whitelisted AST walk.

Not ``eval``.  The model will eventually feed this something it mis-parsed
("2 + 2 apples", "1,000 + 5"), and a tool that segfaults the interpreter on a
malformed expression is a bad neighbour to the loop.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Annotated

from ..errors import ToolError
from .base import tool

# Only these node types are ever constructed while walking the tree.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": lambda *a: sum(a),
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pow": math.pow,
    "hypot": math.hypot,
}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

# Guards against the two ways a calculator tool gets abused: exponential
# blowup (``9**9**9``) and a wall-clock hang.
_MAX_EXPONENT = 128
_MAX_ABS_INTERMEDIATE = 1e308
_NODES = 200


class _Expr(ast.NodeVisitor):
    def __init__(self) -> None:
        self.nodes = 0

    def generic_visit(self, node: ast.AST) -> Any:  # pragma: no cover - guarded by visit
        raise ToolError(
            f"Unsupported expression element: {type(node).__name__}",
            hint="Only numbers, + - * / // % ** and the listed functions are allowed.",
        )

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        self.nodes += 1
        self._tick()
        left, right = self.visit(node.left), self.visit(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > _MAX_EXPONENT:
                raise ToolError(f"Exponent too large (max {_MAX_EXPONENT})")
        if isinstance(node.op, ast.Div | ast.FloorDiv) and right == 0:
            raise ToolError("Division by zero")
        value = _BIN_OPS[type(node.op)](left, right)
        if abs(value) > _MAX_ABS_INTERMEDIATE:
            raise ToolError("Result overflowed")
        return value

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        self.nodes += 1
        self._tick()
        return _UNARY_OPS[type(node.op)](self.visit(node.operand))

    def visit_Constant(self, node: ast.Constant) -> Any:
        self.nodes += 1
        self._tick()
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError(f"Not a number: {node.value!r}")
        return node.value

    def visit_Name(self, node: ast.Name) -> Any:
        self.nodes += 1
        self._tick()
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        if node.id in _FUNCS:
            raise ToolError(f"{node.id} is a function -- call it, e.g. {node.id}(x)")
        raise ToolError(
            f"Unknown name {node.id!r}",
            hint=f"Available constants: {', '.join(sorted(_CONSTS))}.",
        )

    def visit_Call(self, node: ast.Call) -> Any:
        self.nodes += 1
        self._tick()
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ToolError(
                "Unknown function",
                hint=f"Available: {', '.join(sorted(_FUNCS))}.",
            )
        if node.keywords:
            raise ToolError("Keyword arguments are not supported")
        args = [self.visit(a) for a in node.args]
        try:
            value = _FUNCS[node.func.id](*args)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ToolError(f"{node.func.id} failed: {exc}") from exc
        if isinstance(value, complex):
            raise ToolError(f"{node.func.id} produced a complex result")
        return value

    def _tick(self) -> None:
        if self.nodes > _NODES:
            raise ToolError(f"Expression too complex (>{_NODES} nodes)")


@tool(tags=("math",))
def calculator(
    expression: Annotated[
        str,
        "arithmetic expression, e.g. '(128*37)/4 + 19' or 'sqrt(2)*100'; "
        "supports + - * / // % ** and sqrt/abs/round/min/max/sum/log/exp/sin/cos/tan",
    ],
) -> str:
    """Evaluate an arithmetic expression and return the result. Use for any math the model should not do in its head."""
    # Thousands separators ("1,000") are friendly for humans and fatal for
    # eval; strip only commas that group digits into threes, so both
    # ``1,000,000`` and ``min(3,7,1)`` survive unchanged.
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expression.strip().replace("×", "*").replace("÷", "/"))
    if not text:
        raise ToolError("Empty expression")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ToolError(
            f"Cannot parse {expression!r}: {exc.msg}",
            hint="Send a plain expression such as '3 * (4 + 5)'.",
        ) from exc
    value = _Expr().visit(tree)
    if isinstance(value, float):
        shown = f"{value:.10g}"
    else:
        shown = str(value)
    return f"{text} = {shown}"
