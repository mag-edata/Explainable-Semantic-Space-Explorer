"""
transforms package
==================
Vector transformation layer. Receives output from the ``core/`` layer
(embedding matrices) and performs transformation operations such as
projection and clustering.

Dependency direction: ``core/`` → ``transforms/`` (``transforms`` must
never depend on ``ui/``).
"""

from transforms.clustering import ClusterResult, KMeansClusterer
from transforms.projection import ProjectionResult, Projector

__all__ = [
    "KMeansClusterer",
    "ClusterResult",
    "Projector",
    "ProjectionResult",
]
