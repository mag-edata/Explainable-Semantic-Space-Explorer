"""
analysis パッケージ
==================
可視化前処理層。core/ 層の出力を受け取り、
投影・クラスタリングなどの前処理を行う。

依存方向: core/ → analysis/  （analysis は ui/ に依存してはならない）
"""

from analysis.cluster import ClusterResult, KMeansClusterer
from analysis.projection import ProjectionResult, Projector

__all__ = [
    "KMeansClusterer",
    "ClusterResult",
    "Projector",
    "ProjectionResult",
]
