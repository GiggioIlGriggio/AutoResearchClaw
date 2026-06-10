#!/usr/bin/env python3
"""Pre-flight validator for an AutoResearchClaw experiment scaffold.

Usage:
    python preflight.py <scaffold_dir>

Validates the scaffold authoring contract (see SKILL.md and design spec section 9):
  - intra-scaffold imports are package-qualified (from scaffold.x / from .x), not bare
  - the dataset is read via os.environ['RC_DATASET_DIR'], not a hardcoded path
  - SCAFFOLD.md describes the modules' public API

Exit code: 0 if no FAILs, 1 if any FAIL, 2 on usage error. Standard library only.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

FAIL, WARN, OK = "FAIL", "WARN", "OK"

DATA_IO_CALLS = {
    "load", "loadtxt", "genfromtxt", "fromfile", "memmap",
    "open", "read_csv", "read_parquet", "read_json", "read_pickle", "imread",
}

_ABS_RE = re.compile(r"^(/[^/\s]|~/|[A-Za-z]:\\)")

_MANIFEST_STOP = {
    "Args", "Arg", "Returns", "Return", "Yields", "Raises", "Note", "Notes",
    "Example", "Examples", "Dataset", "Model", "Models", "Loader", "Loaders",
    "Util", "Utils", "None", "True", "False", "Tensor", "Path", "List", "Dict",
    "Tuple", "Optional", "Union", "Any", "RC_DATASET_DIR",
}


class Report:
    def __init__(self):
        self.items = []  # list of (category, level, msg)

    def add(self, category, level, msg):
        self.items.append((category, level, msg))

    def of(self, category):
        return [it for it in self.items if it[0] == category]

    @property
    def failed(self):
        return any(level == FAIL for _, level, _ in self.items)


def _call_name(func):
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _str_const(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _looks_absolute(s):
    return bool(_ABS_RE.match(s)) and len(s) > 3


def check_imports(tree, path, siblings, rep):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in siblings:
                    rep.add("imports", FAIL,
                            f"{path.name}:{node.lineno} bare 'import {alias.name}' -> "
                            f"'from scaffold.{alias.name} import ...' or 'from .{alias.name} import ...'")
        elif isinstance(node, ast.ImportFrom):
            if (node.level or 0) > 0:
                continue  # relative import -> fine
            mod = node.module or ""
            if mod.split(".")[0] in siblings:
                rep.add("imports", FAIL,
                        f"{path.name}:{node.lineno} bare 'from {mod} import ...' -> "
                        f"'from scaffold.{mod} import ...' or 'from .{mod} import ...'")


def check_data_access(tree, src, path, rep):
    uses_env = "RC_DATASET_DIR" in src
    does_io = False
    fail_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node.func) in DATA_IO_CALLS:
            does_io = True
            for arg in node.args:
                lit = _str_const(arg)
                if lit and _looks_absolute(lit):
                    fail_lines.add(node.lineno)
                    rep.add("dataset", FAIL,
                            f"{path.name}:{node.lineno} hardcoded path '{lit}' in I/O call -> "
                            f"read os.environ['RC_DATASET_DIR']")
    for node in ast.walk(tree):
        lit = _str_const(node)
        if lit and _looks_absolute(lit) and getattr(node, "lineno", None) not in fail_lines:
            rep.add("dataset", WARN,
                    f"{path.name}:{node.lineno} absolute path literal '{lit}' -> "
                    f"prefer a path under os.environ['RC_DATASET_DIR']")
    if does_io and not uses_env:
        rep.add("dataset", WARN,
                f"{path.name} performs file I/O but never references RC_DATASET_DIR")


def public_api(parsed):
    names = set()
    for tree, _ in parsed:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                    and not node.name.startswith("_"):
                names.add(node.name)
    return names


def check_manifest(scaffold_dir, api, rep):
    mpath = scaffold_dir / "SCAFFOLD.md"
    if not mpath.exists():
        rep.add("manifest", WARN,
                "SCAFFOLD.md not found -- the pipeline would inject only the file tree, so the "
                "LLM sees no description of your API or dataset")
        return
    text = mpath.read_text(encoding="utf-8", errors="replace")
    referenced = set(re.findall(r"`([A-Za-z_]\w*)`", text))
    referenced |= set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", text))
    suspects = {n for n in referenced
                if (n[0].isupper() or "_" in n) and n not in _MANIFEST_STOP}
    for n in sorted(suspects - api):
        rep.add("manifest", WARN,
                f"SCAFFOLD.md mentions '{n}' but no public def named '{n}' exists in the modules")


def main(argv):
    if len(argv) != 2:
        print("usage: preflight.py <scaffold_dir>")
        return 2
    scaffold_dir = Path(argv[1]).expanduser()
    if not scaffold_dir.is_dir():
        print(f"not a directory: {scaffold_dir}")
        return 2
    py_paths = sorted(p for p in scaffold_dir.glob("*.py") if p.name != "__init__.py")
    if not py_paths:
        print(f"no .py modules found in {scaffold_dir}")
        return 2

    siblings = {p.stem for p in py_paths}
    rep = Report()
    parsed = []
    for p in py_paths:
        src = p.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(src, filename=str(p))
        except SyntaxError as e:
            rep.add("syntax", FAIL, f"{p.name}:{e.lineno} syntax error: {e.msg}")
            continue
        parsed.append((tree, p))
        check_imports(tree, p, siblings, rep)
        check_data_access(tree, src, p, rep)
    check_manifest(scaffold_dir, public_api(parsed), rep)

    labels = [
        ("syntax", "modules parse"),
        ("imports", "package-qualified imports"),
        ("dataset", "dataset via RC_DATASET_DIR"),
        ("manifest", "SCAFFOLD.md matches API"),
    ]
    print(f"scaffold: {scaffold_dir}  ({len(py_paths)} module(s))")
    for cat, label in labels:
        items = rep.of(cat)
        if not items:
            print(f"  {OK:<4} {label}")
        else:
            for _, level, msg in items:
                print(f"  {level:<4} {msg}")
    print()
    if rep.failed:
        print("FAILED -- fix the FAIL items above before using this scaffold.")
        return 1
    print("OK -- scaffold satisfies the authoring contract.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
