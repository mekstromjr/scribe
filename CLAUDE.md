# CLAUDE.md

## This repository is public

`meklab/scribe` is developed on a private GitLab instance and push-mirrored, with full history, to
https://github.com/mekstromjr/scribe as a read-only portfolio mirror. Every commit on a protected
branch lands on GitHub within minutes and is cached and indexed from then on.

Treat all of the following as publicly visible: code, docs, tests, fixtures, comments, merge
request titles and descriptions, and commit messages.

- No secrets, tokens, or credentials, including revoked ones. `gitleaks git .` must stay clean.
- No LAN or tailnet IPs. Use hostnames (`*.meklab.net` names are fine) or `<placeholders>`.
- No family names, relatives' emails, or other third-party identities. Use role words: the owner,
  the maintainer, a family member. The maintainer's own name and public email are fine.
- Commits are authored by the maintainer only. Represent anyone else with a neutral placeholder
  identity, never their real name.
- Private operational detail (node names, incident narrative, internal ticket dumps) belongs in the
  private infra repos. Link to them by name; do not reproduce their content here.

History was rewritten once to enforce this (2026-09-08). Check before pushing, not after.

## Workflow

Never commit directly to `main`. Branch, open a merge request, let the pipeline run, then merge.
Releases are semver tags on `main`. GitHub is never pushed to directly.
