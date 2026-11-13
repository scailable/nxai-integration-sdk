"""Tests for the build wiring scan.

The scan is a pure-text CMake mini-tokenizer -- independent of any real CMake
configure -- that REJECTs when a processor is unwired (I1) or, for Python
processors, internally inconsistent (I2-I5). These tests build small fixture trees
under ``tempfile.mkdtemp()`` rather than touching the real repo, except for the one
integration test that scans this repo's actual root.
"""

import json
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import build_wiring_scan as scan


_PYTHON_CMAKE = """\
set(PROCESSOR_NAME "{name}")
add_custom_target(${{PROCESSOR_NAME}} ALL
    COMMAND ${{PIP_EXECUTABLE}} install -r requirements.txt
)
install(
    PROGRAMS ${{CMAKE_CURRENT_BINARY_DIR}}/${{PROCESSOR_NAME}}
    DESTINATION ${{{dest_var}}}
    COMPONENT {component}
)
"""

_CPP_CMAKE = """\
project({name} CXX C)
add_executable({name} main.cpp)
install(TARGETS {name}
    DESTINATION ${{INSTALL_DEST_POSTPROCESSORS}}
    COMPONENT {name}
)
"""


def _write_root_cmake(root, add_subdirectory_lines):
    body = "\n".join(add_subdirectory_lines)
    with open(os.path.join(root, "CMakeLists.txt"), "w") as handle:
        handle.write(body + "\n")


def _write_python_processor(root, name, dest_var=None, component=None, json=True,
                             py_file=True, requirements=True):
    processor_dir = os.path.join(root, name)
    os.makedirs(processor_dir, exist_ok=True)
    dest_var = dest_var or (
        "INSTALL_DEST_PREPROCESSORS" if name.startswith("preprocessor-")
        else "INSTALL_DEST_POSTPROCESSORS"
    )
    component = component if component is not None else "${PROCESSOR_NAME}"
    with open(os.path.join(processor_dir, "CMakeLists.txt"), "w") as handle:
        handle.write(_PYTHON_CMAKE.format(name=name, dest_var=dest_var, component=component))
    if json:
        with open(os.path.join(processor_dir, "external_postprocessors.json"), "w") as handle:
            handle.write("{}")
    if py_file:
        with open(os.path.join(processor_dir, f"{name}.py"), "w") as handle:
            handle.write("# processor\n")
    if requirements:
        with open(os.path.join(processor_dir, "requirements.txt"), "w") as handle:
            handle.write("msgpack\n")
    return processor_dir


def _write_cpp_processor(root, name, json=True):
    processor_dir = os.path.join(root, name)
    os.makedirs(processor_dir, exist_ok=True)
    with open(os.path.join(processor_dir, "CMakeLists.txt"), "w") as handle:
        handle.write(_CPP_CMAKE.format(name=name))
    if json:
        with open(os.path.join(processor_dir, "external_postprocessors.json"), "w") as handle:
            handle.write("{}")
    return processor_dir


class TestI1Wiring(unittest.TestCase):
    def test_wired_and_migrated_passes(self):
        root = tempfile.mkdtemp()
        _write_python_processor(root, "postprocessor-python-example")
        _write_root_cmake(root, ["add_subdirectory(postprocessor-python-example)"])
        result = scan.scan_tree(root)
        self.assertEqual(result.findings, [])
        self.assertEqual(result.verdict, "PASS")

    def test_migrated_but_unwired_is_a_finding(self):
        """Regression: three processors shipped with a registration
        JSON but no root add_subdirectory line."""
        root = tempfile.mkdtemp()
        _write_python_processor(root, "postprocessor-python-events-example")
        _write_root_cmake(root, [])
        result = scan.scan_tree(root)
        self.assertEqual(result.verdict, "REJECT")
        self.assertEqual(len(result.findings), 1)
        self.assertEqual(result.findings[0].invariant, "I1")
        self.assertEqual(result.findings[0].processor, "postprocessor-python-events-example")

    def test_commented_out_add_subdirectory_still_counts_as_unwired(self):
        """A naive regex would treat a commented-out add_subdirectory as real
        wiring (a false PASS) -- the scan must not."""
        root = tempfile.mkdtemp()
        _write_python_processor(root, "postprocessor-python-example")
        _write_root_cmake(root, ["# add_subdirectory(postprocessor-python-example)"])
        result = scan.scan_tree(root)
        self.assertEqual(result.verdict, "REJECT")
        self.assertEqual(result.findings[0].invariant, "I1")

    def test_unmigrated_dir_with_no_json_and_no_wiring_passes(self):
        root = tempfile.mkdtemp()
        _write_python_processor(root, "postprocessor-python-anpr-example", json=False)
        _write_root_cmake(root, [])
        result = scan.scan_tree(root)
        self.assertEqual(result.verdict, "PASS")

    def test_wired_but_unmigrated_is_a_finding(self):
        root = tempfile.mkdtemp()
        _write_python_processor(root, "postprocessor-python-anpr-example", json=False)
        _write_root_cmake(root, ["add_subdirectory(postprocessor-python-anpr-example)"])
        result = scan.scan_tree(root)
        self.assertEqual(result.verdict, "REJECT")
        self.assertEqual(result.findings[0].invariant, "I1")

    def test_duplicate_wiring_is_a_finding(self):
        root = tempfile.mkdtemp()
        _write_python_processor(root, "postprocessor-python-example")
        _write_root_cmake(root, [
            "add_subdirectory(postprocessor-python-example)",
            "add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-example)",
        ])
        result = scan.scan_tree(root)
        self.assertEqual(result.verdict, "REJECT")
        self.assertEqual(result.findings[0].invariant, "I1")

    def test_cmake_current_source_dir_prefixed_subdirectory_counts_as_wired(self):
        root = tempfile.mkdtemp()
        _write_python_processor(root, "postprocessor-python-example")
        _write_root_cmake(root, [
            "add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-example)",
        ])
        result = scan.scan_tree(root)
        self.assertEqual(result.verdict, "PASS")

    def test_bracket_commented_out_add_subdirectory_still_counts_as_unwired(self):
        """Same false-PASS risk as line comments, but for #[[...]] bracket
        comments -- a naive line-comment-only stripper would miss this."""
        root = tempfile.mkdtemp()
        _write_python_processor(root, "postprocessor-python-example")
        _write_root_cmake(root, [
            "#[[\nadd_subdirectory(postprocessor-python-example)\n]]",
        ])
        result = scan.scan_tree(root)
        self.assertEqual(result.verdict, "REJECT")
        self.assertEqual(result.findings[0].invariant, "I1")

    def test_quoted_string_resembling_a_command_call_is_not_mistaken_for_one(self):
        """A quoted string like "Program Files (x86)" must not be parsed as a call
        to a command named "files" -- the scan masks string content, not just
        comments, before looking for command names."""
        root = tempfile.mkdtemp()
        _write_python_processor(root, "postprocessor-python-example")
        _write_root_cmake(root, [
            'message(STATUS "Installs under Program Files (x86)")',
            "add_subdirectory(postprocessor-python-example)",
        ])
        result = scan.scan_tree(root)
        self.assertEqual(result.findings, [])
        self.assertEqual(result.verdict, "PASS")


class TestPythonOnlyInvariants(unittest.TestCase):
    def test_cpp_processor_only_checks_i1(self):
        """C++ examples have no PROCESSOR_NAME, requirements.txt, or PROGRAMS
        install -- I2-I5 must not fire for them."""
        root = tempfile.mkdtemp()
        _write_cpp_processor(root, "postprocessor-cpp-objects-example")
        _write_root_cmake(root, ["add_subdirectory(postprocessor-cpp-objects-example)"])
        result = scan.scan_tree(root)
        self.assertEqual(result.findings, [])
        self.assertEqual(result.verdict, "PASS")

    def test_i2_processor_name_mismatch_is_a_finding(self):
        root = tempfile.mkdtemp()
        processor_dir = _write_python_processor(root, "postprocessor-python-example")
        with open(os.path.join(processor_dir, "CMakeLists.txt"), "w") as handle:
            handle.write(_PYTHON_CMAKE.format(
                name="postprocessor-python-wrong-name",
                dest_var="INSTALL_DEST_POSTPROCESSORS",
                component="${PROCESSOR_NAME}",
            ))
        _write_root_cmake(root, ["add_subdirectory(postprocessor-python-example)"])
        result = scan.scan_tree(root)
        self.assertEqual(result.verdict, "REJECT")
        self.assertEqual(result.findings[0].invariant, "I2")

    def test_i2_missing_py_file_is_a_finding(self):
        root = tempfile.mkdtemp()
        _write_python_processor(root, "postprocessor-python-example", py_file=False)
        _write_root_cmake(root, ["add_subdirectory(postprocessor-python-example)"])
        result = scan.scan_tree(root)
        self.assertEqual(result.verdict, "REJECT")
        self.assertEqual(result.findings[0].invariant, "I2")

    def test_i3_missing_requirements_txt_is_a_finding(self):
        root = tempfile.mkdtemp()
        _write_python_processor(root, "postprocessor-python-example", requirements=False)
        _write_root_cmake(root, ["add_subdirectory(postprocessor-python-example)"])
        result = scan.scan_tree(root)
        self.assertEqual(result.verdict, "REJECT")
        self.assertEqual(result.findings[0].invariant, "I3")

    def test_i4_preprocessor_wired_to_postprocessor_dest_is_a_finding(self):
        """Regression for a near-miss: a preprocessor's CMakeLists.txt copied
        from a postprocessor example, with INSTALL_DEST_POSTPROCESSORS never
        flipped to INSTALL_DEST_PREPROCESSORS."""
        root = tempfile.mkdtemp()
        _write_python_processor(
            root, "preprocessor-python-tensor-example",
            dest_var="INSTALL_DEST_POSTPROCESSORS",
        )
        _write_root_cmake(root, ["add_subdirectory(preprocessor-python-tensor-example)"])
        result = scan.scan_tree(root)
        self.assertEqual(result.verdict, "REJECT")
        self.assertEqual(result.findings[0].invariant, "I4")

    def test_i4_preprocessor_wired_to_preprocessor_dest_passes(self):
        root = tempfile.mkdtemp()
        _write_python_processor(root, "preprocessor-python-tensor-example")
        _write_root_cmake(root, ["add_subdirectory(preprocessor-python-tensor-example)"])
        result = scan.scan_tree(root)
        self.assertEqual(result.verdict, "PASS")

    def test_i5_component_mismatch_is_a_finding(self):
        root = tempfile.mkdtemp()
        _write_python_processor(
            root, "postprocessor-python-example", component="${PROJECT_NAME}",
        )
        _write_root_cmake(root, ["add_subdirectory(postprocessor-python-example)"])
        result = scan.scan_tree(root)
        self.assertEqual(result.verdict, "REJECT")
        self.assertEqual(result.findings[0].invariant, "I5")

    def test_i5_literal_component_matching_processor_name_passes(self):
        root = tempfile.mkdtemp()
        _write_python_processor(
            root, "postprocessor-python-example",
            component="postprocessor-python-example",
        )
        _write_root_cmake(root, ["add_subdirectory(postprocessor-python-example)"])
        result = scan.scan_tree(root)
        self.assertEqual(result.verdict, "PASS")


class TestFormatReport(unittest.TestCase):
    def test_clean_report_ends_in_pass(self):
        result = scan.ScanResult(findings=[])
        report = scan.format_report(result)
        self.assertTrue(report.endswith("VERDICT: PASS"))

    def test_dirty_report_lists_findings_and_ends_in_reject(self):
        result = scan.ScanResult(findings=[
            scan.Finding("I1", "postprocessor-python-example", "not wired"),
        ])
        report = scan.format_report(result)
        self.assertIn("I1: postprocessor-python-example: not wired", report)
        self.assertTrue(report.endswith("VERDICT: REJECT"))


class TestActualRepoTree(unittest.TestCase):
    def test_real_sdk_root_passes(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        result = scan.scan_tree(repo_root)
        self.assertEqual(result.findings, [], scan.format_report(result))
        self.assertEqual(result.verdict, "PASS")

    def test_registration_files_do_not_share_a_log_filepath(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        owners = {}
        for processor in sorted(os.listdir(repo_root)):
            if not processor.startswith(("postprocessor-", "preprocessor-")):
                continue
            for name in os.listdir(os.path.join(repo_root, processor)):
                if not re.match(r"external_p.*processors\.json$", name):
                    continue
                with open(os.path.join(repo_root, processor, name), encoding="utf-8") as handle:
                    entries = json.load(handle)
                for entry in entries:
                    owners.setdefault(entry.get("LogFilepath"), []).append(processor)
        owners.pop(None, None)
        shared = {path: who for path, who in owners.items() if len(who) > 1}
        self.assertEqual(shared, {})


if __name__ == "__main__":
    unittest.main()
