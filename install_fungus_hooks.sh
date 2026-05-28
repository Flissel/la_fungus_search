#!/bin/sh
# install_fungus_hooks.sh — point every VibeMind git repo at the central
# post-commit hook so a commit in ANY repo (root or any submodule, however
# deep) triggers an incremental fungus-search reindex.
#
# Mechanism: git's core.hooksPath. Each repo gets core.hooksPath set to the
# single shared hooks dir (.githooks/ at the Vibemind_V1 root). One hook file,
# zero drift, and re-running this script picks up newly-added submodules.
#
# Idempotent — safe to run repeatedly. Run after `git submodule update`.
#
# Usage:
#     sh vibemind-os/la-fungus-search/install_fungus_hooks.sh
#
# Plan: C:\Users\User\.claude\plans\plan-das-mal-soft-wren.md

set -e

ROOT="C:/Users/User/Desktop/Vibemind_V1"
HOOKS_DIR="${ROOT}/.githooks"

if [ ! -f "${HOOKS_DIR}/post-commit" ]; then
  echo "ERROR: ${HOOKS_DIR}/post-commit not found — central hook missing" >&2
  exit 1
fi

# Make the hook executable (git needs the exec bit on POSIX; harmless on Win).
chmod +x "${HOOKS_DIR}/post-commit" 2>/dev/null || true

installed=0
set_hookspath() {
  # $1 = repo working-dir
  repo="$1"
  if [ ! -e "${repo}/.git" ]; then
    return 0
  fi
  ( cd "${repo}" && git config core.hooksPath "${HOOKS_DIR}" )
  echo "  [ok] core.hooksPath set: ${repo}"
  installed=$((installed + 1))
}

echo "=== Installing central fungus-search post-commit hook ==="
echo "Central hook: ${HOOKS_DIR}/post-commit"
echo ""

# 1. Root repo
set_hookspath "${ROOT}"

# 2. vibemind-os (has its own real .git dir)
set_hookspath "${ROOT}/vibemind-os"

# 3. Every sub-submodule of vibemind-os (auto-discovered)
if [ -d "${ROOT}/vibemind-os" ]; then
  ( cd "${ROOT}/vibemind-os" && git submodule status 2>/dev/null | awk '{print $2}' ) \
  | while read -r sub; do
      [ -n "${sub}" ] || continue
      set_hookspath "${ROOT}/vibemind-os/${sub}"
    done
fi

echo ""
echo "Done. Every commit in any of these repos now triggers an incremental"
echo "reindex (background, GPU, ~15-30s; ~0s if nothing indexable changed)."
echo ""
echo "Verify a single repo with:  git -C <repo> config core.hooksPath"
