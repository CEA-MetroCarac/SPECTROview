# Purpose

This is a reusable Markdown template for filing SPECTROview bug reports.

---

# Bug Report

**Date:** YYYY-MM-DD
**Reporter:** [Name / GitHub handle]
**SPECTROview Version:** [e.g., 26.29.3]
**Operating System:** [macOS 14.4 / Windows 11 / Ubuntu 22.04]

---

## Summary

One-line description of the bug.

> **Example:** Application crashes when loading a `.wdf` file with more than 10,000 spectra.

---

## Bug Type

- [ ] Crash / unhandled exception
- [ ] Incorrect result or calculation
- [ ] UI / display issue
- [ ] Data loss
- [ ] Performance degradation
- [ ] Other: ___________

---

## Affected Component

- [ ] File Loading workspace
- [ ] Spectra workspace
- [ ] Graphs workspace
- [ ] Maps workspace
- [ ] MVA workspace
- [ ] AI Agent
- [ ] VBF engine
- [ ] Settings
- [ ] Other: ___________

---

## Steps to Reproduce

List the exact steps to trigger the bug.

1. Open SPECTROview.
2. Go to [Workspace].
3. Load the file `[filename.ext]`.
4. Click [Button / Action].
5. Observe: [what happens].

---

## Expected Behaviour

Describe what should have happened.

> **Example:** The file should load successfully and display spectra in the Spectra workspace.

---

## Actual Behaviour

Describe what actually happened, including any error message.

> **Example:** The application freezes for 30 seconds, then crashes with an unhandled exception:
> ```
> AttributeError: 'NoneType' object has no attribute 'shape'
> File "spectroview/model/m_io.py", line 227, in load_wdf_map
> ```

---

## Error Log / Traceback

Paste the full Python traceback or relevant log output.

```
[paste traceback here]
```

---

## Reproducibility

- [ ] Always (100%)
- [ ] Sometimes (%)
- [ ] Only with specific files

---

## File / Data

Can you share the file that triggers the bug?

- [ ] Yes — attach to issue
- [ ] No — contains confidential data
- [ ] Not applicable

---

## Additional Context

Any other information that may be relevant (hardware, Python version, installed packages).

```
Python version: 3.XX
PySide6 version: X.X.X
numpy version: X.X.X
```
