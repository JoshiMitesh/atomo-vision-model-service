"""Slave Node — distributed real-time video detection system.

Current scope: Person Detection only (see /CLAUDE.md, /AGENTS.md, /specs.md).

Run with:
    python -m slave.main
from the repository root (the parent of this package), so that this
package is imported as ``slave`` and never sits on ``sys.path`` itself.
That matters because this package contains a ``slave.logging`` module —
running scripts from *inside* this directory would shadow the standard
library ``logging`` module.
"""
