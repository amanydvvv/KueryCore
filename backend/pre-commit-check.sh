#!/bin/sh
# ============================================================
# KueryCore — Pre-commit hook
# Prevents committing code that references an alembic migration
# that hasn't been tracked in git yet (the root cause of the
# Render crash-loop in Oct 2026).
# ============================================================

# Check for untracked migration files
UNTRACKED=$(git ls-files --others --exclude-standard backend/alembic/versions/)
if [ -n "$UNTRACKED" ]; then
  printf "\n\033[0;31m❌  PRE-COMMIT BLOCKED: Untracked migration file(s) found:\033[0m\n"
  printf "%s\n" "$UNTRACKED"
  printf "\n\033[0;33m  Stage them first:\033[0m\n"
  printf "  git add %s\n" "$UNTRACKED"
  printf "\n  Then re-run your commit.\n\n"
  exit 1
fi

exit 0
