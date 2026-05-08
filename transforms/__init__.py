"""
transforms パッケージ
====================
ベクトル変換層。core/ 層の出力（埋め込み行列）を受け取り、
投影・クラスタリングなどの変換操作を行う。

依存方向: core/ → transforms/ （transforms は ui/ に依存してはならない）
"""

from transforms.clustering import ClusterResult, KMeansClusterer
from transforms.projection import ProjectionResult, Projector

__all__ = [
    "KMeansClusterer",
    "ClusterResult",
    "Projector",
    "ProjectionResult",
]
