import ast
from typing import Dict, Set


class UnsafeLogicExpression(Exception):
    pass


_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.UnaryOp,
    ast.Name,
    ast.Load,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Constant,
)


def extract_logic_names(expr: str) -> Set[str]:
    """Return variable names used in a boolean emit expression."""
    if not expr or not expr.strip():
        return set()

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise UnsafeLogicExpression(f"Invalid expression syntax: {exc}") from exc

    names: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise UnsafeLogicExpression(
                f"Disallowed node: {type(node).__name__}"
            )
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names


def safe_eval_bool_expr(expr: str, values: Dict[str, bool]) -> bool:
    """Safely evaluate a boolean expression over judge-step variables."""
    if not expr or not expr.strip():
        raise UnsafeLogicExpression("Empty boolean expression.")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise UnsafeLogicExpression(f"Invalid expression syntax: {exc}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise UnsafeLogicExpression(
                f"Disallowed node: {type(node).__name__}"
            )
        if isinstance(node, ast.Name) and node.id not in values:
            raise UnsafeLogicExpression(f"Unknown variable: {node.id}")

    return _eval_node(tree.body, values)


def _eval_node(node: ast.AST, values: Dict[str, bool]) -> bool:
    if isinstance(node, ast.Name):
        return bool(values[node.id])

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return node.value
        raise UnsafeLogicExpression("Only boolean constants are allowed.")

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval_node(node.operand, values)

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(_eval_node(v, values) for v in node.values)
        if isinstance(node.op, ast.Or):
            return any(_eval_node(v, values) for v in node.values)

    raise UnsafeLogicExpression(f"Unsupported expression: {ast.dump(node)}")
