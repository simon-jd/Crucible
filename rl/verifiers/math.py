"""Bounded rational-expression verifier; model strings never enter eval/parse_expr.

Grammar: decimal integers, + - * / ^, parentheses/braces, x/y, sqrt/\\sqrt, pi/\\pi, \\frac.
Powers are literal integers in [-8, 8]. No implicit multiplication or other functions.
Symbolic equality is rational-function equality (removable poles ignored).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl.verifiers.types import Verdict

MAX_INPUT = 4096


class MathSyntaxError(ValueError):
    """A stable, deliberately bounded grammar failure reason."""


def _answer(text: str) -> str:
    if not isinstance(text, str) or not text or len(text) > MAX_INPUT:
        raise MathSyntaxError("input_limit")
    boxes = list(re.finditer(r"\\boxed\s*\{", text))
    if text.count(r"\boxed") != len(boxes):
        raise MathSyntaxError("malformed_box")
    if boxes:
        # Require every box to balance, then use the last final answer.
        answers = []
        for box in boxes:
            depth, start = 1, box.end()
            for end in range(start, len(text)):
                depth += (text[end] == "{") - (text[end] == "}")
                if depth == 0:
                    answers.append(text[start:end])
                    break
            else:
                raise MathSyntaxError("malformed_box")
        return answers[-1]
    return text


def _parse(text):
    import sympy as sp

    text = _answer(text)
    pattern = re.compile(r"\s*(\\frac|\\sqrt|\\pi|sqrt|pi|\d+(?:\.\d+)?|[xy+*/^(){}-])")
    tokens, pos = [], 0
    while pos < len(text):
        if not text[pos:].strip():
            break
        match = pattern.match(text, pos)
        if not match:
            raise MathSyntaxError("unsupported_syntax")
        token = match.group(1)
        if len(token) > 32:
            raise MathSyntaxError("number_limit")
        tokens.append(token)
        pos = match.end()
    if not tokens or len(tokens) > 96:
        raise MathSyntaxError("token_limit")
    index, operations = 0, 0

    def take(expected=None):
        nonlocal index
        if index >= len(tokens) or (expected is not None and tokens[index] != expected):
            raise MathSyntaxError("incomplete_expression")
        value = tokens[index]
        index += 1
        return value

    def expression(minimum=0, depth=0):
        nonlocal operations
        if depth > 12:
            raise MathSyntaxError("depth_limit")
        token = take()
        if token in ("+", "-"):
            value = expression(3, depth + 1) * (-1 if token == "-" else 1)
        elif token in ("(", "{"):
            value = expression(0, depth + 1)
            take(")" if token == "(" else "}")
        elif token == r"\frac":
            take("{")
            numerator = expression(0, depth + 1)
            take("}")
            take("{")
            denominator = expression(0, depth + 1)
            take("}")
            if denominator == 0:
                raise MathSyntaxError("division_by_zero")
            value = numerator / denominator
        elif token in ("sqrt", r"\sqrt"):
            opening = take()
            if opening not in ("(", "{"):
                raise MathSyntaxError("sqrt_group_required")
            radicand = expression(0, depth + 1)
            take(")" if opening == "(" else "}")
            if radicand.is_negative:
                raise MathSyntaxError("complex_not_supported")
            value = sp.sqrt(radicand)
        elif token in ("pi", r"\pi"):
            value = sp.pi
        elif token in ("x", "y"):
            value = sp.Symbol(token)
        elif re.fullmatch(r"\d+(?:\.\d+)?", token):
            value = sp.Rational(token)
        else:
            raise MathSyntaxError("unsupported_syntax")
        priority = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 4}
        while index < len(tokens) and priority.get(tokens[index], -1) >= minimum:
            operator = take()
            operations += 1
            if operations > 32:
                raise MathSyntaxError("operation_limit")
            if operator == "^":
                negative = index < len(tokens) and tokens[index] == "-"
                if negative:
                    take("-")
                power = take()
                if not power.isdigit() or int(power) > 8:
                    raise MathSyntaxError("power_limit")
                exponent = int(power) * (-1 if negative else 1)
                if value == 0 and exponent <= 0:
                    raise MathSyntaxError("undefined_power")
                value = value**exponent
                if index < len(tokens) and tokens[index] == "^":
                    raise MathSyntaxError("power_chain")
                continue
            right = expression(priority[operator] + 1, depth + 1)
            if operator == "+":
                value += right
            elif operator == "-":
                value -= right
            elif operator == "*":
                value *= right
            else:
                if right == 0:
                    raise MathSyntaxError("division_by_zero")
                value /= right
            if sp.count_ops(value) > 128:
                raise MathSyntaxError("complexity_limit")
        return value

    result = expression()
    if index != len(tokens):
        raise MathSyntaxError("trailing_syntax")
    if result.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise MathSyntaxError("nonfinite")
    return result


def verify_math(prediction: str, reference: str) -> Verdict:
    from rl.verifiers.types import Verdict

    try:
        _answer(prediction)
        _answer(reference)
        result = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).resolve()), "--worker"],
            input=json.dumps([_answer(prediction), _answer(reference)], ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if result.returncode:
            return Verdict(False, "math_worker_failed")
        payload = json.loads(result.stdout)
        return Verdict(payload["correct"] is True, payload["reason"])
    except subprocess.TimeoutExpired:
        return Verdict(False, "math_timeout")
    except MathSyntaxError as error:
        return Verdict(False, "invalid_math:" + str(error))
    except (ValueError, OSError, KeyError):
        return Verdict(False, "invalid_math")


def _worker():
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    try:
        import sympy as sp

        prediction, reference = json.loads(sys.stdin.read(4 * MAX_INPUT + 128))
        left, right = _parse(prediction), _parse(reference)
        difference = sp.simplify(sp.cancel(left - right))
        correct = difference == 0
        result = {
            "correct": bool(correct),
            "reason": "equivalent" if correct else "not_equivalent_or_unknown",
        }
    except MathSyntaxError as error:
        result = {"correct": False, "reason": "invalid_math:" + str(error)}
    except Exception as error:  # noqa: BLE001 - bounded worker must fail closed
        result = {"correct": False, "reason": "invalid_math:" + type(error).__name__}
    print(json.dumps(result))


if __name__ == "__main__":
    _worker()
