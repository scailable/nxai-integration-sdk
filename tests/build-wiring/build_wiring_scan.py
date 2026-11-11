"""Build wiring scan: guard that every migrated processor is reachable from the root
``CMakeLists.txt``.

Because CMake+Nuitka builds are slow, both e2e test layers run processors
*interpreted*: they invoke ``<processor>/<processor>.py`` through the
Python interpreter directly and never touch a CMake target. So a processor whose
``CMakeLists.txt`` is never ``add_subdirectory``'d from the root has no build target
at all, and neither CMake configure nor either test harness notices. This happened
repeatedly during the AIM v5 migration.

The discriminator for "this processor should be wired" is a colocated
``external_p*processors.json`` registration file (I1). Given that a processor is a
Python one (as opposed to the two C++ examples, which only need I1), I2-I5 also
guard the exact copy-paste slip that has happened before: flipping ``INSTALL_DEST_POSTPROCESSORS`` to
``INSTALL_DEST_PREPROCESSORS`` (or vice versa) when a new processor's
``CMakeLists.txt`` was copied from the nearest existing example.

This is a mini-tokenizer, not a CMake evaluator: it strips line comments,
``#[[...]]`` bracket comments and quoted strings' delimiters (but not their content),
then extracts ``add_subdirectory`` / ``set`` / ``install`` calls. The only way this
scan can fail its purpose is a false PASS -- e.g. a raw regex mistaking a
commented-out ``add_subdirectory`` for real wiring -- which is why comments are
genuinely stripped rather than pattern-matched around.

A PASS here does not mean CMake configure passes, does not mean Nuitka builds, and
does not mean the artifact works -- those are separate tiers of testing. This only asserts that a build target *exists* and is *wired up*.
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List


@dataclass
class Finding:
    invariant: str
    processor: str
    detail: str


@dataclass
class ScanResult:
    findings: List[Finding] = field(default_factory=list)

    @property
    def verdict(self):
        return "REJECT" if self.findings else "PASS"


def scan_tree(root):
    """Scan the processor directories under ``root`` and return a :class:`ScanResult`.

    ``root`` is the SDK repo root (the directory containing the root
    ``CMakeLists.txt``).
    """
    findings = []
    root_cmake_path = os.path.join(root, "CMakeLists.txt")
    with open(root_cmake_path, encoding="utf-8") as handle:
        root_commands = list(_iter_commands(handle.read()))
    wired_dirs = [
        _basename_of_subdirectory_arg(args)
        for name, args in root_commands
        if name == "add_subdirectory"
    ]

    for processor in sorted(_iter_processor_dirs(root)):
        cmake_path = os.path.join(root, processor, "CMakeLists.txt")
        with open(cmake_path, encoding="utf-8") as handle:
            commands = list(_iter_commands(handle.read()))
        has_json = _has_registration_json(os.path.join(root, processor))
        wired_count = wired_dirs.count(processor)
        findings.extend(_check_i1(processor, has_json, wired_count))

        if _is_cpp_shaped(commands):
            continue
        findings.extend(_check_python_invariants(root, processor, commands))

    return ScanResult(findings=findings)


def format_report(result):
    """Render a human-readable report ending in the overall verdict line."""
    lines = ["Build wiring scan:"]
    if result.findings:
        for f in result.findings:
            lines.append(f"  {f.invariant}: {f.processor}: {f.detail}")
    else:
        lines.append("  every processor is wired and internally consistent")
    lines.append(f"VERDICT: {result.verdict}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        help="SDK repo root (defaults to two levels above this script)",
    )
    args = parser.parse_args(argv)

    result = scan_tree(args.root)
    print(format_report(result))
    return 1 if result.verdict == "REJECT" else 0


def _iter_processor_dirs(root):
    """Yield the names of top-level ``postprocessor-*`` / ``preprocessor-*`` dirs
    that have a ``CMakeLists.txt`` (i.e. every processor, migrated or not)."""
    for name in os.listdir(root):
        if not (name.startswith("postprocessor-") or name.startswith("preprocessor-")):
            continue
        if os.path.isfile(os.path.join(root, name, "CMakeLists.txt")):
            yield name


def _has_registration_json(processor_dir):
    return any(
        re.match(r"external_p.*processors\.json$", name)
        for name in os.listdir(processor_dir)
    )


def _is_cpp_shaped(commands):
    """C++ examples use ``add_executable`` + ``install(TARGETS ...)``; Python
    examples use ``set(PROCESSOR_NAME ...)`` + ``install(PROGRAMS ...)``. Only the
    former shape is exempt from I2-I5."""
    return any(name == "add_executable" for name, _args in commands)


def _basename_of_subdirectory_arg(args):
    tokens = _tokenize_args(args)
    path = tokens[0] if tokens else ""
    path = path.replace("${CMAKE_CURRENT_SOURCE_DIR}/", "")
    return path.rstrip("/").split("/")[-1]


def _check_i1(processor, has_json, wired_count):
    if has_json and wired_count == 0:
        return [Finding("I1", processor, "has registration JSON but no root add_subdirectory")]
    if has_json and wired_count > 1:
        return [Finding("I1", processor, f"wired {wired_count} times (expected exactly 1)")]
    if not has_json and wired_count > 0:
        return [Finding("I1", processor, "wired from root but has no registration JSON")]
    return []


def _check_python_invariants(root, processor, commands):
    findings = []
    table = _build_set_table(commands)
    processor_dir = os.path.join(root, processor)

    processor_name = table.get("PROCESSOR_NAME")
    if processor_name != processor:
        findings.append(Finding(
            "I2", processor,
            f"PROCESSOR_NAME is {processor_name!r}, expected {processor!r}",
        ))
    elif not os.path.isfile(os.path.join(processor_dir, f"{processor_name}.py")):
        findings.append(Finding("I2", processor, f"{processor_name}.py does not exist"))

    if not os.path.isfile(os.path.join(processor_dir, "requirements.txt")):
        findings.append(Finding("I3", processor, "requirements.txt does not exist"))

    expected_dest_var = (
        "INSTALL_DEST_PREPROCESSORS" if processor.startswith("preprocessor-")
        else "INSTALL_DEST_POSTPROCESSORS"
    )
    install_calls = [(name, args) for name, args in commands if name == "install"]
    for _name, args in install_calls:
        tokens = _tokenize_args(args)
        dest = _arg_after_keyword(tokens, "DESTINATION")
        if dest is not None and dest != f"${{{expected_dest_var}}}":
            findings.append(Finding(
                "I4", processor, f"DESTINATION is {dest!r}, expected '${{{expected_dest_var}}}'",
            ))

        component = _arg_after_keyword(tokens, "COMPONENT")
        if component is not None:
            resolved = _resolve(component, table)
            if resolved != processor_name:
                findings.append(Finding(
                    "I5", processor,
                    f"COMPONENT resolves to {resolved!r}, expected {processor_name!r}",
                ))

    return findings


def _build_set_table(commands):
    table = {}
    for name, args in commands:
        if name != "set":
            continue
        tokens = _tokenize_args(args)
        if len(tokens) >= 2:
            table[tokens[0]] = tokens[1]
    return table


def _resolve(token, table):
    match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", token)
    if match:
        return table.get(match.group(1))
    return token


def _arg_after_keyword(tokens, keyword):
    for i, tok in enumerate(tokens):
        if tok == keyword and i + 1 < len(tokens):
            return tokens[i + 1]
    return None


def _strip_comments(text):
    """Remove line comments and ``#[[...]]``-style bracket comments, leaving quoted
    strings (and everything else) intact."""
    out = []
    i, n = 0, len(text)
    in_string = False
    while i < n:
        c = text[i]
        if in_string:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            out.append(c)
            i += 1
            continue
        if c == "#":
            bracket = re.match(r"#(\[=*\[)", text[i:])
            if bracket:
                open_tag = bracket.group(1)
                eq = open_tag[1:-1]
                close_tag = "]" + eq + "]"
                end = text.find(close_tag, i + len(open_tag) + 1)
                i = n if end == -1 else end + len(close_tag)
                continue
            end = text.find("\n", i)
            i = n if end == -1 else end
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _mask_strings(text):
    """Return a same-length copy of ``text`` with quoted string content (and its
    delimiters) replaced by spaces, so command-name matching never looks inside a
    string literal."""
    out = []
    i, n = 0, len(text)
    in_string = False
    while i < n:
        c = text[i]
        if in_string:
            if c == "\\" and i + 1 < n:
                out.append("  ")
                i += 2
                continue
            if c == '"':
                in_string = False
            out.append(" ")
            i += 1
            continue
        if c == '"':
            in_string = True
            out.append(" ")
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


_COMMAND_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _iter_commands(text):
    """Yield ``(command_name_lower, raw_args_string)`` for every top-level CMake
    command call in ``text``, after stripping comments.

    Command names are matched against a string-masked copy of the text (quoted
    string content blanked out, same length) so that a quoted string such as
    ``"Program Files (x86)"`` can never be mistaken for a call to a command named
    ``files``. The masked copy is only used to find match positions; argument
    extraction still reads the real (comment-stripped) text.
    """
    cleaned = _strip_comments(text)
    masked = _mask_strings(cleaned)
    for match in _COMMAND_RE.finditer(masked):
        start = match.end()
        depth = 1
        i = start
        in_string = False
        n = len(cleaned)
        while i < n and depth > 0:
            c = cleaned[i]
            if in_string:
                if c == "\\":
                    i += 2
                    continue
                if c == '"':
                    in_string = False
            elif c == '"':
                in_string = True
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            i += 1
        yield match.group(1).lower(), cleaned[start:i - 1]


def _tokenize_args(args):
    """Split a raw CMake argument string into tokens, honoring quoted strings as a
    single token with their delimiters and escapes removed."""
    tokens = []
    i, n = 0, len(args)
    while i < n:
        c = args[i]
        if c.isspace():
            i += 1
            continue
        if c == '"':
            j = i + 1
            buf = []
            while j < n and args[j] != '"':
                if args[j] == "\\" and j + 1 < n:
                    buf.append(args[j + 1])
                    j += 2
                    continue
                buf.append(args[j])
                j += 1
            tokens.append("".join(buf))
            i = j + 1
            continue
        j = i
        while j < n and not args[j].isspace():
            j += 1
        tokens.append(args[i:j])
        i = j
    return tokens


if __name__ == "__main__":
    sys.exit(main())
