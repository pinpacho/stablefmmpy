"""
tree.py — FMMNode, QuadTree: adaptive hierarchical partitioning.

References:
  [HK]  HelmholtzKernel2D.pdf — Michelle, Ou, Xia; preprint 2024
  [M2D] Multipole2D.pdf — Ou, Michelle, Xia; SIAM J. Matrix Anal. Appl. 46(1), 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .core import PointSet


# ---------------------------------------------------------------------------
# FMMNode — single node in the adaptive quad-tree
# ---------------------------------------------------------------------------

@dataclass
class FMMNode:
    """A single box in the FMM quad-tree.

    Stores geometry, point indices, tree links, and the working vectors
    v^(l), t^(l), u^(l) from Algorithm 4.1 [HK §4; M2D §4].
    """
    center: complex
    radius: float
    level: int
    box_id: int

    src_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    tgt_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    parent: Optional['FMMNode'] = field(default=None, repr=False)
    children: List['FMMNode'] = field(default_factory=list, repr=False)
    interaction_list: List['FMMNode'] = field(default_factory=list, repr=False)
    near_list: List['FMMNode'] = field(default_factory=list, repr=False)

    v_vec: Optional[np.ndarray] = field(default=None, repr=False)
    t_vec: Optional[np.ndarray] = field(default=None, repr=False)
    u_vec: Optional[np.ndarray] = field(default=None, repr=False)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_well_separated(self, other: 'FMMNode', tau: float = 0.6) -> bool:
        """True iff r_i + r_j <= tau * |c_i - c_j| [HK §1 separation condition]."""
        return (self.radius + other.radius <= tau * abs(self.center - other.center))


# ---------------------------------------------------------------------------
# QuadTree — adaptive hierarchical partitioning
# ---------------------------------------------------------------------------

class QuadTree:
    """Adaptive quad-tree over source and target point sets.

    Each node splits when it contains more than N0 points. Splitting uses
    the Re/Im midpoint of the bounding box (quad-tree).
    """

    def __init__(self):
        self.root: Optional[FMMNode] = None
        self._nodes: List[FMMNode] = []
        self._node_counter = 0

    def build(self, src_pts: np.ndarray, tgt_pts: np.ndarray,
              tau: float = 0.6, N0: int = 32) -> None:
        """Build the quad-tree for src_pts (sources) and tgt_pts (targets)."""
        all_pts = np.concatenate([src_pts, tgt_pts])
        n_src = len(src_pts)

        center = complex(np.mean(all_pts))
        radius = float(np.max(np.abs(all_pts - center))) * 1.01
        src_all = np.arange(n_src, dtype=int)
        tgt_all = np.arange(len(tgt_pts), dtype=int)  # LOCAL: 0..n_tgt-1

        self.root = FMMNode(center=center, radius=radius,
                            level=1, box_id=self._new_id(),
                            src_idx=src_all, tgt_idx=tgt_all)
        self._nodes = [self.root]

        stack = [self.root]
        while stack:
            node = stack.pop()
            n_pts = len(node.src_idx) + len(node.tgt_idx)
            if n_pts > N0:
                children = self._split(node, src_pts, tgt_pts, n_src)
                if children:
                    node.children = children
                    for c in children:
                        self._nodes.append(c)
                    stack.extend(children)

        self._build_lists(tau)

    def _new_id(self) -> int:
        self._node_counter += 1
        return self._node_counter

    def _split(self, node: FMMNode, src_pts: np.ndarray,
               tgt_pts: np.ndarray, n_src: int) -> List[FMMNode]:
        """Split node into up to 4 children using Re/Im quadrant membership.

        The strict < vs >= convention guarantees every point lands in exactly
        one quadrant (no duplicates, no gaps). [HK §4, tree construction]
        """
        cx, cy = node.center.real, node.center.imag
        half = node.radius / 2.0

        child_centers = [
            complex(cx - half, cy + half),  # NW
            complex(cx + half, cy + half),  # NE
            complex(cx - half, cy - half),  # SW
            complex(cx + half, cy - half),  # SE
        ]
        quadrant_signs = [(-1, +1), (+1, +1), (-1, -1), (+1, -1)]

        def _select(pts_arr, idx_arr, sre, sim):
            if len(idx_arr) == 0:
                return np.array([], dtype=int)
            pts = pts_arr[idx_arr]
            re_mask = (pts.real >= cx) if sre > 0 else (pts.real < cx)
            im_mask = (pts.imag >= cy) if sim > 0 else (pts.imag < cy)
            return idx_arr[re_mask & im_mask]

        children = []
        for cc, (sre, sim) in zip(child_centers, quadrant_signs):
            s_idx = _select(src_pts, node.src_idx, sre, sim)
            t_idx = _select(tgt_pts, node.tgt_idx, sre, sim)
            if len(s_idx) + len(t_idx) == 0:
                continue
            child = FMMNode(center=cc, radius=half,
                            level=node.level + 1, box_id=self._new_id(),
                            src_idx=s_idx, tgt_idx=t_idx, parent=node)
            children.append(child)

        return children

    def _build_lists(self, tau: float) -> None:
        """Populate interaction_list (M2L) and near_list (P2P) for each leaf."""
        leaves = [n for n in self._nodes if n.is_leaf()]
        for node in leaves:
            for other in leaves:
                if other is node:
                    continue
                if node.is_well_separated(other, tau):
                    node.interaction_list.append(other)
                else:
                    node.near_list.append(other)

    def by_level(self) -> Dict[int, List[FMMNode]]:
        """Return dict mapping level -> list of nodes at that level."""
        result: Dict[int, List[FMMNode]] = {}
        for n in self._nodes:
            result.setdefault(n.level, []).append(n)
        return result

    def postorder(self) -> List[FMMNode]:
        """Bottom-up traversal: children before parent."""
        result = []
        stack = [self.root] if self.root else []
        while stack:
            node = stack.pop()
            result.append(node)
            stack.extend(node.children)
        return list(reversed(result))

    def preorder(self) -> List[FMMNode]:
        """Top-down traversal: parent before children."""
        result = []
        stack = [self.root] if self.root else []
        while stack:
            node = stack.pop()
            result.append(node)
            stack.extend(reversed(node.children))
        return result

    def max_level(self) -> int:
        return max((n.level for n in self._nodes), default=1)

    def visualize(self, src_pts: np.ndarray, tgt_pts: np.ndarray) -> object:
        """Plot quad-tree boxes colored by level. Returns matplotlib Figure."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=(8, 8))

        cmap = plt.cm.Blues
        max_lv = self.max_level()
        for node in self._nodes:
            color = cmap(0.3 + 0.6 * node.level / max(max_lv, 1))
            alpha = 0.15 + 0.25 * node.level / max(max_lv, 1)
            sq = patches.Rectangle(
                (node.center.real - node.radius, node.center.imag - node.radius),
                2 * node.radius, 2 * node.radius,
                linewidth=0.8, edgecolor='navy', facecolor=color, alpha=alpha)
            ax.add_patch(sq)

        ax.scatter(src_pts.real, src_pts.imag, c='red', s=8,
                   label='Sources (Y)', zorder=5, alpha=0.7)
        ax.scatter(tgt_pts.real, tgt_pts.imag, c='blue', s=8,
                   label='Targets (X)', zorder=5, alpha=0.7)
        ax.set_aspect('equal')
        ax.legend(fontsize=10)
        ax.set_title('Particion jerarquica (QuadTree FMM)', fontsize=12)
        ax.set_xlabel('Re(z)')
        ax.set_ylabel('Im(z)')
        fig.tight_layout()
        return fig
