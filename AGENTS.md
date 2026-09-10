# Working agreement

- Do not launch or remain attached to full GPU training from Codex.
- Put every GPU task in a standalone script that the user can run without an
  active Codex session.
- Use Codex for diagnostics, instrumentation, and structural model or training
  changes, not as a substitute for a hyperparameter optimizer.
- Analyze completed training artifacts and gather evidence before proposing
  changes.
- Preserve subject-disjoint dataset splits and never modify manifests as part of
  an experiment unless the user explicitly requests it.
