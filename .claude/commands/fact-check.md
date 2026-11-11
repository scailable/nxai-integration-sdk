Verify that CLAUDE.md accurately describes the current state of the repository. Report any discrepancies and offer to fix them.

## Steps

1. **Read** `CLAUDE.md` at the repo root.

2. **Post-processor count** — count directories matching `postprocessor-*/` at the repo root. Compare to the number stated in CLAUDE.md.

3. **Preprocessor count** — count directories matching `preprocessor-*/`. Compare to the number stated in CLAUDE.md.

4. **File and directory references** — for every path or filename mentioned in CLAUDE.md, verify it exists on disk:
   - All `.claude/docs/*.md` filenames (check exact names, not just the directory)
   - Every `postprocessor-*/` and `preprocessor-*/` directory name mentioned in the "Picking an example to copy" section
   - Top-level items: `nxai-utilities/`, `message_processing_utils/`, `example_models/`, `CMakeLists.txt`, `build_all.sh`, `build_docker.sh`

5. **Root-CMake reachability** is not this skill's job — run the build wiring scan instead: `python tests/build-wiring/build_wiring_scan.py --root .` — the same check as CTest case `build-wiring-scan`.

6. **"Picking an example" mappings** — for every directory name listed in the "Picking an example to copy" section of CLAUDE.md, confirm the directory exists.

7. **Install paths** — verify the Linux install paths stated in CLAUDE.md match the `INSTALL_DEST_POSTPROCESSORS` and `INSTALL_DEST_PREPROCESSORS` default values in root `CMakeLists.txt`.

8. **Print** a full checklist:
   ```
   ✅ Post-processor count: 14 (matches)
   ✅ Preprocessor count: 3 (matches)
   ❌ docs/ at repo root: does not exist — actual location is .claude/docs/
   ✅ postprocessor-python-example: exists
   ...
   ```

9. For every ❌ or ⚠️ item, **propose the specific fix** (exact text to change in CLAUDE.md or CMakeLists.txt) and offer to apply it.
