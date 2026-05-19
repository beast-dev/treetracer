"""Midpoint rooting for newick trees.

Used by ``mcc/_subprocess_worker.py`` when the source distmat was
computed with ``is_rooted=False`` (MrBayes / RevBayes unrooted output).
The MCC algorithm picks the best tree from the posterior in a
rooting-agnostic way; we then need to root the chosen tree somewhere
sensible for display. Midpoint rooting is the standard default —
roots on the centre of the diameter (longest leaf-to-leaf path), which
is well-defined for any tree with branch lengths.

Pure-Python implementation. No external dependencies. ~200 LoC.

Algorithm (standard "two-BFS" approach):

  1. Parse the newick into an N-ary node tree with branch lengths.
  2. BFS from any leaf u0 → find the farthest leaf u.
  3. BFS from u → find the farthest leaf v. The path u→v is the
     diameter D.
  4. Walk u→v summing branch lengths until we've covered D/2 — this
     locates the midpoint, which sits either on an edge or coincides
     with an existing node.
  5. Insert a new root node at the midpoint (or use the coincident
     existing node), and reorient the tree so that all edges point
     away from the new root.
  6. Emit the rerooted tree as newick.

Edge cases handled:
  * Fewer than 3 leaves → returns the input unchanged (no meaningful
    midpoint).
  * Midpoint lands exactly on an existing node → that node becomes the
    new root; no synthetic node insertion.
  * Polytomies (3+ children) → algorithm treats nodes as N-ary so they
    pass through naturally.
  * Zero-length branches → contribute zero distance; algorithm robust.
  * Missing branch lengths → treated as 0.0 (BEAST / MrBayes / RevBayes
    always emit branch lengths so this only fires on hand-edited input).
  * BEAST inline annotations ``[&rate=…]`` — stripped before parsing.
    The user said dropping annotations is fine; midpoint rooting only
    needs topology + branch lengths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class _Node:
    """One node of an N-ary phylogenetic tree.

    ``children`` is a list of (child, edge_length) tuples — keeping
    the edge length on the parent side simplifies the reroot pass
    (no need to mutate the child's own length to flip orientation).
    """
    name: Optional[str] = None
    children: List[Tuple["_Node", float]] = field(default_factory=list)
    parent: Optional["_Node"] = None
    length_to_parent: float = 0.0  # length of the edge from parent to this node


# ─── Annotation stripping ─────────────────────────────────────────────
# BEAST writes ``[&rate=0.05,…]`` on internal/leaf nodes and edges.
# Standard newick doesn't allow ``[...]`` — strip before parsing.
_ANN_RE = re.compile(r"\[[^\]]*\]")


def _strip_annotations(newick: str) -> str:
    return _ANN_RE.sub("", newick)


# ─── Newick parser ────────────────────────────────────────────────────
# Recursive-descent. Recognises:
#   newick      := subtree ';'
#   subtree     := '(' subtree (',' subtree)* ')' [name] [':' length]
#                | name [':' length]
#   name        := any chars except (),:;[]
#   length      := float

class _NewickReader:
    def __init__(self, s: str):
        self.s = s
        self.i = 0

    def _peek(self) -> str:
        return self.s[self.i] if self.i < len(self.s) else ""

    def _read_subtree(self) -> _Node:
        node = _Node()
        if self._peek() == "(":
            self.i += 1
            while True:
                child = self._read_subtree()
                length = self._read_length()
                child.length_to_parent = length
                child.parent = node
                node.children.append((child, length))
                if self._peek() == ",":
                    self.i += 1
                    continue
                if self._peek() == ")":
                    self.i += 1
                    break
                raise ValueError(
                    f"newick parse: expected ',' or ')' at pos {self.i}; "
                    f"saw {self._peek()!r}"
                )
        name = self._read_name()
        if name:
            node.name = name
        return node

    def _read_name(self) -> str:
        start = self.i
        while (self.i < len(self.s)
               and self.s[self.i] not in "(),:;"):
            self.i += 1
        return self.s[start:self.i].strip()

    def _read_length(self) -> float:
        if self._peek() != ":":
            return 0.0
        self.i += 1
        start = self.i
        # Branch lengths can be scientific notation, signed, decimal.
        while (self.i < len(self.s)
               and (self.s[self.i].isdigit()
                    or self.s[self.i] in ".+-eE")):
            self.i += 1
        token = self.s[start:self.i]
        try:
            return float(token) if token else 0.0
        except ValueError:
            return 0.0

    def read(self) -> _Node:
        root = self._read_subtree()
        # Consume trailing ';' if present.
        if self._peek() == ";":
            self.i += 1
        return root


def _parse_newick(newick: str) -> _Node:
    return _NewickReader(_strip_annotations(newick).strip()).read()


# ─── Tree traversal helpers ───────────────────────────────────────────

def _all_nodes(root: _Node) -> List[_Node]:
    out: List[_Node] = []
    stack = [root]
    while stack:
        n = stack.pop()
        out.append(n)
        for c, _ in n.children:
            stack.append(c)
    return out


def _leaves(root: _Node) -> List[_Node]:
    return [n for n in _all_nodes(root) if not n.children]


def _neighbours(node: _Node) -> List[Tuple[_Node, float]]:
    """Edges incident to ``node`` in the UNDIRECTED sense — both
    children and (if present) the parent."""
    edges = list(node.children)
    if node.parent is not None:
        edges.append((node.parent, node.length_to_parent))
    return edges


def _farthest_leaf(start: _Node) -> Tuple[_Node, float]:
    """BFS from ``start`` treating the tree as undirected; return the
    farthest leaf and its total branch-length distance."""
    visited = {id(start)}
    stack: List[Tuple[_Node, float]] = [(start, 0.0)]
    best_leaf, best_dist = start, 0.0
    while stack:
        node, d = stack.pop()
        if not node.children and d > best_dist:
            best_leaf, best_dist = node, d
        for nbr, w in _neighbours(node):
            if id(nbr) not in visited:
                visited.add(id(nbr))
                stack.append((nbr, d + w))
    return best_leaf, best_dist


def _path(a: _Node, b: _Node) -> List[Tuple[_Node, _Node, float]]:
    """Return the sequence of (from_node, to_node, length) edges along
    the unique tree path from a to b, in order."""
    # Build the path via BFS storing predecessors.
    pred: dict[int, Tuple[_Node, float]] = {id(a): (None, 0.0)}  # type: ignore
    queue: List[_Node] = [a]
    while queue:
        cur = queue.pop(0)
        if cur is b:
            break
        for nbr, w in _neighbours(cur):
            if id(nbr) not in pred:
                pred[id(nbr)] = (cur, w)
                queue.append(nbr)
    # Walk back from b → a, collect, reverse.
    path: List[Tuple[_Node, _Node, float]] = []
    cur = b
    while cur is not a:
        prev, w = pred[id(cur)]
        path.append((prev, cur, w))
        cur = prev
    path.reverse()
    return path


# ─── Re-rooting ───────────────────────────────────────────────────────

def _reroot_at_node(new_root: _Node) -> _Node:
    """Reorient the tree so ``new_root`` is the root. Flips parent /
    child relationships along the path from ``new_root`` to the old
    root; everything else stays put.

    Edges keep their lengths — only direction changes.
    """
    if new_root.parent is None:
        return new_root  # already the root

    # Capture the chain BEFORE mutating anything. The flip loop below
    # rewrites parent pointers along this chain, so following them
    # live would oscillate (each flip sets ``parent.parent = cur``,
    # which would be revisited on the next iteration if we walked
    # ``cur = cur.parent`` mid-flip).
    chain: List[Tuple[_Node, _Node, float]] = []
    cur = new_root
    while cur.parent is not None:
        chain.append((cur, cur.parent, cur.length_to_parent))
        cur = cur.parent

    # new_root is now the actual root.
    new_root.parent = None
    new_root.length_to_parent = 0.0

    # Walk the captured chain, flipping each (child, parent) edge:
    # parent becomes a child of child; everything else parent had
    # stays where it was (subtrees off the path don't need to move).
    for child, parent, edge_len in chain:
        parent.children = [(c, w) for c, w in parent.children if c is not child]
        child.children.append((parent, edge_len))
        parent.parent = child
        parent.length_to_parent = edge_len
    return new_root


def _split_edge(parent: _Node, child: _Node, offset_from_parent: float) -> _Node:
    """Insert a new node on the edge from ``parent`` to ``child`` at
    distance ``offset_from_parent`` from parent. Returns the new node.

    Lengths split: parent → new is ``offset_from_parent``,
                   new    → child is ``original_length - offset``.
    """
    # Find and remove the edge in parent's children list.
    original_len = None
    new_children = []
    for c, w in parent.children:
        if c is child:
            original_len = w
        else:
            new_children.append((c, w))
    if original_len is None:
        raise ValueError("_split_edge: child not found in parent's children")
    parent.children = new_children

    offset = max(0.0, min(offset_from_parent, original_len))
    upper_len = offset
    lower_len = original_len - offset

    new_node = _Node()
    new_node.parent = parent
    new_node.length_to_parent = upper_len
    parent.children.append((new_node, upper_len))

    child.parent = new_node
    child.length_to_parent = lower_len
    new_node.children.append((child, lower_len))
    return new_node


# ─── Newick emitter ───────────────────────────────────────────────────

def _emit_newick(root: _Node) -> str:
    parts: List[str] = []
    _emit_node(root, parts)
    parts.append(";")
    return "".join(parts)


def _emit_node(node: _Node, out: List[str]) -> None:
    if node.children:
        out.append("(")
        for i, (c, w) in enumerate(node.children):
            if i:
                out.append(",")
            _emit_node(c, out)
            # Always emit branch length for non-root edges; readers
            # vary in tolerance for missing lengths.
            out.append(":")
            out.append(_fmt_length(w))
        out.append(")")
    if node.name:
        out.append(node.name)


def _fmt_length(x: float) -> str:
    # Match BEAST's typical %g formatting; avoid scientific notation
    # for "normal" branch lengths because some downstream parsers
    # don't handle ``1e-05`` style. Round to 8 decimal places.
    if x == 0.0:
        return "0"
    s = f"{x:.8f}".rstrip("0").rstrip(".")
    return s or "0"


# ─── Public entry point ───────────────────────────────────────────────

def midpoint_root_newick(newick: str) -> str:
    """Re-root the tree at its midpoint and return a new newick string.

    See module docstring for the algorithm and edge-case handling.
    Returns the input unchanged if the tree has fewer than 3 leaves
    (no meaningful midpoint exists).
    """
    root = _parse_newick(newick)
    leaves = _leaves(root)
    if len(leaves) < 3:
        return newick

    # Two-BFS diameter discovery.
    u, _ = _farthest_leaf(leaves[0])
    v, diameter = _farthest_leaf(u)
    half = diameter / 2.0

    # Walk u → v summing edge lengths; stop when we've covered ``half``.
    path = _path(u, v)
    cumulative = 0.0
    target_edge: Optional[Tuple[_Node, _Node, float]] = None
    offset_into_edge = 0.0
    for parent_side, child_side, w in path:
        if cumulative + w >= half:
            target_edge = (parent_side, child_side, w)
            offset_into_edge = half - cumulative
            break
        cumulative += w
    if target_edge is None:
        # All edges summed to less than half — shouldn't happen, but
        # fall back to leaving the tree as is.
        return newick

    parent_side, child_side, edge_len = target_edge

    # The path's "parent_side" / "child_side" labels reflect traversal
    # order from u → v, not the actual parent-child pointers (the path
    # walks up and down through the LCA). Find the true parent-child
    # direction by checking which is the actual parent.
    if child_side.parent is parent_side:
        true_parent, true_child = parent_side, child_side
        offset_from_true_parent = offset_into_edge
    elif parent_side.parent is child_side:
        true_parent, true_child = child_side, parent_side
        offset_from_true_parent = edge_len - offset_into_edge
    else:
        # The two are siblings of a common parent? Shouldn't happen in
        # a tree path. Fall back.
        return newick

    # If the midpoint lands exactly on a node, reuse that node as the
    # new root — no synthetic insertion needed. Treat anything within
    # 1e-12 of an endpoint as coincident.
    if offset_from_true_parent <= 1e-12:
        new_root_node = true_parent
    elif edge_len - offset_from_true_parent <= 1e-12:
        new_root_node = true_child
    else:
        new_root_node = _split_edge(
            true_parent, true_child, offset_from_true_parent,
        )

    new_root = _reroot_at_node(new_root_node)
    return _emit_newick(new_root)
