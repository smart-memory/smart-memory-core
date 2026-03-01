"""
SmartMemory version information.

Version is read dynamically from package metadata or VERSION file.
Single source of truth: VERSION file
"""

from pathlib import Path

def _get_version() -> str:
    """Get version from VERSION file or package metadata."""
    try:
        # For installed package, read from metadata
        from importlib.metadata import version
        return version("smartmemory-core")
    except Exception:
        # For development, read from VERSION file
        try:
            version_file = Path(__file__).parent.parent / "VERSION"
            with open(version_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            # Fallback if VERSION file not found
            return "unknown"

__version__ = _get_version()

# Parse version info
try:
    __version_info__ = tuple(int(x) for x in __version__.split("."))
except Exception:
    __version_info__ = (0, 1, 6)

# Version history:
# 0.3.0 - Decision Memory: Decision model, DecisionManager, DecisionQueries, DecisionExtractor, edge schemas (PRODUCED, DERIVED_FROM, SUPERSEDES, CONTRADICTS, INFLUENCES)
# 0.2.6 - SSG integration, user signature support, documentation updates, DRY version management (VERSION file as single source of truth)
# 0.1.14 - Tenant isolation bug fix
# 0.1.13 - Tenant isolation bug fix
# 0.1.12 - Tenant isolation improvements
# 0.1.11 - Version bump and maintenance
# 0.1.10 - Minor fix to grounder plugin, publish script improvements
# 0.1.9 - Add dspy to requirements
# 0.1.8 - README overhaul: fixed all code snippets to use public API, verified evolvers, added "In Progress" section, removed internal imports
# 0.1.7 - Updated README, removed ChromaDB references, fixed PyPI deployment
# 0.1.6 - Production PyPI deployment setup
# 0.1.5 - Complete bi-temporal implementation: version tracking, temporal search, relationship queries, bi-temporal joins, performance optimizations
# 0.1.4 - Bi-temporal queries: time-travel, audit trails, version history, rollback
# 0.1.3 - Zettelkasten system with wikilink support, documentation, examples, CLI
# 0.1.2 - ChromaDB optional, Python 3.10+ requirement, version externalized
# 0.1.1 - Plugin system with security
