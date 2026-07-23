# Purpose

This file defines plotting-specific behavioral rules for the SPECTROview AI Agent. These rules apply whenever calling `plot_graph`, `update_graph`, or `delete_graph`. Field-level mechanics — defaults, spatial-axis mapping, multi-style syntax, filter quoting — live in `prompts/plotting.md`; this file covers what NOT to do.

---

# Instructions

These rules govern how the AI agent creates, modifies, and deletes graphs.

---

# Rules

- **Never add decorative elements** (extra annotations, watermarks, legends) that the user did not request.
- **Never generate a plot referencing columns or data that do not exist** in the loaded DataFrame — validate first.
- **Never change an axis the user did not mention.** On `update_graph`, omit `x`, `y`, and `plot_style` unless the user explicitly asked to change them — omitted arguments keep their current value.
- **Never answer "group by / colour by / split by `<column>`" with `x`.** That is the `z` (hue) argument; `x` stays as it is. Putting the grouping column on `x` throws away the axis the user was actually looking at.
- **Never use `seaborn`** or reference any plotting library — the agent never generates standalone code (see `rules/general.md` Safety); every plot goes through `plot_graph`/`update_graph`.
- **Never set spine/border styling on `wafer` plots** — a wafer plot uses only the left spine by design; the app hides the top/right/bottom borders and applies this automatically. Do not pass `spines_visible` for a wafer plot (it can wrongly re-enable all four borders).

---

# Constraints

- If a plotting request conflicts with these rules, explain the conflict in plain text rather than silently ignoring it.
