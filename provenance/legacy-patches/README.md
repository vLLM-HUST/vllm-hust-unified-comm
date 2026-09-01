# Legacy patch archive

This directory preserves the original contribution commits behind Unified
Communication.  The files are unmodified `git format-patch` exports, so author,
date, commit message, trailers, and exact diff remain reviewable.

- `core-pr-42/`: original commits from legacy core PR #42.
- `core-pr-168/`: original commits from legacy core PR #168.

These patches are migration evidence, not patches to apply blindly to a current
vLLM checkout.  Runtime-independent code should be moved into this package;
changes that still require vLLM hooks must be represented as an explicit host
contract and tested against a declared vLLM version range.

