---
description: Preserve original audio names and checkpoint compatibility when changing workflow exports
applyTo: "src/**/*.rs,tests/test_api_integration.py"
---

# Workflow exports

## Learnings

- Final filenames come from the original audio stem stored in `ProjectStatus.export_stem`, not the project directory or the latest replacement input. Capture the filename before canonicalization; rebinding identical content changes `input` but must preserve the stored stem. The optional field keeps schema-2 legacy states readable.
- Use the shared `export_paths` helpers for pipeline writes, CLI output, API results, hashes and archives. If an audio stem collides with an internal JSON filename (for example, `workflow.mp3`), place both exports in `exports/` rather than overwriting workflow state.
- An export-format change invalidates only the export fingerprint. Archive legacy `transcript.*` using the previous paths before generating named files; preserve valid ASR and review checkpoints.
- Build export temporary names by appending `.partial` to the complete destination filename. Extension replacement exposed a directory-path failure for the valid dot-stem audio name `..mp3`; retain coverage for actual writes, Unicode/dotted names, collisions, rebinding and legacy migration.
