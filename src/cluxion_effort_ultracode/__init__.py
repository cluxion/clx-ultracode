"""Effort-Ultracode: portable multi-agent orchestration with unanimous-consensus debate."""

from importlib.metadata import PackageNotFoundError, version

from cluxion_effort_ultracode.core import ConsensusEngine, ConsensusResult

try:
    __version__ = version("cluxion-agentplugin-effort-ultracode")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.13"

__all__ = ["ConsensusEngine", "ConsensusResult"]
