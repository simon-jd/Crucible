"""Conservative abuse signals, never the sandbox/security boundary.

Flags may reject benign uses. Obfuscation can evade them; hidden grading and
runtime isolation must remain effective when this detector returns [].
"""

import ast
import re


def detect_hacks(solution: str, run_log: str) -> list[str]:
    flags = set()
    if not isinstance(solution, str) or len(solution) > 65536:
        return ["source_limit"]
    try:
        tree = ast.parse(solution)
    except (SyntaxError, ValueError, RecursionError):
        return ["invalid_python"]
    pending = [(tree, 0)]
    nodes = 0
    while pending:
        node, depth = pending.pop()
        nodes += 1
        if nodes > 4096 or depth > 64:
            return ["source_complexity"]
        pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                aliases[alias.asname or alias.name] = (node.module or "") + "." + alias.name

    def name(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return name(node.value) + "." + node.attr
        return ""

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            called = name(node.func)
            if called in ("sys.exit", "os._exit", "exit", "quit"):
                flags.add("early_exit")
            if (
                called == "print"
                and node.args
                and all(isinstance(arg, ast.Constant) for arg in node.args)
            ):
                flags.add("literal_output")
            if (
                called in ("setattr", "delattr")
                and node.args
                and name(node.args[0]).startswith(("unittest", "builtins"))
            ):
                flags.add("monkeypatch_tests")
            if called in ("open", "io.open") and node.args:
                path = node.args[0]
                mode = (
                    node.args[1]
                    if len(node.args) > 1
                    else next((kw.value for kw in node.keywords if kw.arg == "mode"), None)
                )
                if (
                    isinstance(path, ast.Constant)
                    and isinstance(path.value, str)
                    and re.search(r"(?:tests?[/_.]|/tests)", path.value)
                    and isinstance(mode, ast.Constant)
                    and isinstance(mode.value, str)
                    and any(ch in mode.value for ch in "wax+")
                ):
                    flags.add("write_tests")
            if called.endswith(
                (".write_text", ".write_bytes", ".unlink", ".rename")
            ) and "test" in ast.unparse(node.func):
                flags.add("write_tests")
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                name(target).startswith(("unittest.", "builtins.assert", "builtins.__import__"))
                for target in targets
            ):
                flags.add("monkeypatch_tests")
        if isinstance(node, ast.Raise) and node.exc is not None:
            target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if name(target) == "SystemExit":
                flags.add("early_exit")
    if re.search(
        r"(read-only file system|permission denied).*tests?|tests?.*(read-only file system|permission denied)",
        run_log[:65536],
        re.IGNORECASE | re.DOTALL,
    ):
        flags.add("write_tests")
    return sorted(flags)
