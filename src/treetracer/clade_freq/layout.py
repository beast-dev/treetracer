"""Pure-Python newick parser and rectangular tree layout engine.

Produces (x, y) coordinates for every node in a rooted, time-calibrated
tree, suitable for drawing with Plotly go.Scatter.

Layout convention
-----------------
- x-axis  = cumulative branch length from the root (time axis).
  Root is at x=0; tips are at their total root-to-tip distance.
- y-axis  = tip rank (0, 1, 2, ...) assigned by an in-order traversal.
  Internal nodes sit at the mean y of their children.

The tree is drawn as an L-shaped (rectangular/cladogram) layout:
  - A horizontal segment from a node's x to its parent's x at the node's y.
  - A vertical segment from the topmost to the bottommost child y at the
    parent's x.

NEXUS translate map
-------------------
BEAST / MrBayes newick strings use integer tokens as tip labels.
Pass the translate map {token: taxon_name} and it is applied during parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Node:
    name:     str            = ""
    length:   float          = 0.0       # branch length to parent
    children: list["Node"]   = field(default_factory=list)
    # Layout fields filled by _assign_layout():
    x:        float          = 0.0       # cumulative root-to-node distance
    y:        float          = 0.0       # tip-rank based vertical position
    is_tip:   bool           = False


# ---------------------------------------------------------------------------
# Newick tokeniser
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
      \[  [^\]]*  \]          # NEXUS metadata comment [...]  — skip
    | ( [(),;] )              # structural punctuation
    | ( [^(),:;\[\]\s]+ )     # label (tip name or internal label)
    | : \s* ( [0-9Ee.+\-]+ ) # branch length after ':'
    """,
    re.VERBOSE,
)


_COMMENT_RE = re.compile(r"\[[^\]]*\]")

def _tokenise(newick: str) -> list[str]:
    """Return a flat list of meaningful tokens, stripping all [...] comments."""
    # Strip all [...] annotation blocks before tokenising.
    clean = _COMMENT_RE.sub("", newick)
    tokens: list[str] = []
    for m in _TOKEN_RE.finditer(clean):
        punct, label, length = m.group(1), m.group(2), m.group(3)
        if punct:
            tokens.append(punct)
        elif label:
            tokens.append(label)
        elif length:
            tokens.append(":" + length)
    return tokens


# ---------------------------------------------------------------------------
# Recursive descent parser
# ---------------------------------------------------------------------------

def _parse_node(tokens: list[str], pos: int) -> tuple[Node, int]:
    """Parse one node (and its subtree) starting at tokens[pos].

    Returns (node, new_pos).
    """
    node = Node()

    if tokens[pos] == "(":
        pos += 1  # consume '('
        while True:
            child, pos = _parse_node(tokens, pos)
            node.children.append(child)
            if tokens[pos] == ",":
                pos += 1  # consume ','
            elif tokens[pos] == ")":
                pos += 1  # consume ')'
                break
            else:
                raise ValueError(f"Unexpected token at {pos}: {tokens[pos]!r}")
        # Optional internal label after ')'.
        # Must not consume a branch-length token (starts with ':').
        if (pos < len(tokens)
                and tokens[pos] not in ("(", ")", ",", ";")
                and not tokens[pos].startswith(":")):
            node.name = tokens[pos]
            pos += 1
    else:
        # Leaf: consume label
        node.name = tokens[pos]
        node.is_tip = True
        pos += 1

    # Optional branch length
    if pos < len(tokens) and tokens[pos].startswith(":"):
        node.length = float(tokens[pos][1:])
        pos += 1

    return node, pos


def _parse_newick(newick: str) -> Node:
    """Parse a newick string and return the root Node."""
    newick = newick.strip().rstrip(";").strip()
    tokens = _tokenise(newick)
    root, _ = _parse_node(tokens, 0)
    return root


# ---------------------------------------------------------------------------
# Translate map application
# ---------------------------------------------------------------------------

def _apply_translate(node: Node, translate: dict[str, str]) -> None:
    """Recursively replace integer tip tokens with taxon names."""
    if node.is_tip and node.name in translate:
        node.name = translate[node.name]
    for child in node.children:
        _apply_translate(child, translate)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _assign_x(node: Node, parent_x: float = 0.0) -> None:
    """Assign cumulative root-to-node x positions."""
    node.x = parent_x + node.length
    for child in node.children:
        _assign_x(child, node.x)


def _assign_y(node: Node, counter: list[int]) -> None:
    """Assign y positions: tips get integer ranks, internals get child mean."""
    if not node.children:
        node.y = float(counter[0])
        counter[0] += 1
    else:
        for child in node.children:
            _assign_y(child, counter)
        node.y = sum(c.y for c in node.children) / len(node.children)

def _ladderize(node: Node) -> None:
    """Recursively sort children so the subtree with fewer tips comes first.
    
    This gives an ascending staircase layout (smallest clade on top).
    Does not change topology.
    """
    for child in node.children:
        _ladderize(child)
    if node.children:
        node.children.sort(key=lambda n: _count_tips(n))


def _count_tips(node: Node) -> int:
    """Count the number of tips in a subtree."""
    if not node.children:
        return 1
    return sum(_count_tips(c) for c in node.children)

def _assign_layout(root: Node) -> None:
    _ladderize(root)
    _assign_x(root, 0.0)
    _assign_y(root, [0])


def _collect_nodes(node: Node) -> list[Node]:
    result = [node]
    for child in node.children:
        result.extend(_collect_nodes(child))
    return result


# ---------------------------------------------------------------------------
# NEXUS parsing
# ---------------------------------------------------------------------------

_TRANSLATE_RE = re.compile(
    r"Translate\s*\n(.*?)\n\s*;", re.IGNORECASE | re.DOTALL
)
#_TREE_LINE_RE = re.compile(
#    r"^\s*tree\s+\S+.*?=\s*(\(.*)", re.IGNORECASE | re.MULTILINE
#)
_TREE_LINE_RE = re.compile(
    r"^\s*tree\s+\S+\s*(?:\[[^\]]*\]\s*)*=\s*(?:\[[^\]]*\]\s*)?(\(.*)",
    re.IGNORECASE | re.MULTILINE
)


def _strip_outer_quotes(label: str) -> str:
    if len(label) >= 2 and label[0] == label[-1] and label[0] in ("'", '"'):
        return label[1:-1]
    return label


def parse_nexus(nexus_bytes: bytes) -> tuple[Node, dict[str, str]]:
    """Parse NEXUS bytes into a laid-out root Node and translate map."""
    text = nexus_bytes.decode("utf-8", errors="replace")

    # Extract translate map.
    translate: dict[str, str] = {}
    tm = _TRANSLATE_RE.search(text)
    if tm:
        for line in tm.group(1).splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                translate[parts[0]] = _strip_outer_quotes(parts[1])

    # Find the tree line by scanning line by line.
    # Handles any annotation between tree name and '=', and any annotation
    # between '=' and the opening '(' of the newick string.
    newick = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.lower().startswith("tree "):
            continue
        # # Find the '=' sign
        # eq_idx = stripped.find("=")
        # if eq_idx == -1:
        #     continue
        # # Everything after '=' is the newick, possibly preceded by annotations
        # after_eq = stripped[eq_idx + 1:].strip()
        # # Skip any leading [...] annotations (e.g. [&R], [&lnCladeCred=...])
        # while after_eq.startswith("["):
        #     close = after_eq.find("]")
        #     if close == -1:
        #         break
        #     after_eq = after_eq[close + 1:].strip()
        # # Now after_eq should start with '('
        # if after_eq.startswith("("):
        #     newick = after_eq.rstrip(";").strip()
        #     break
        # Find the '=' at bracket depth 0, ignoring '=' inside [...] blocks.
        depth = 0
        eq_pos = -1
        for i, c in enumerate(stripped):
            if c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
            elif c == '=' and depth == 0:
                eq_pos = i
                break
        if eq_pos == -1:
            continue
        after_eq = stripped[eq_pos + 1:].strip()
        # Skip any leading [...] annotations after '=' (e.g. [&R])
        while after_eq.startswith("["):
            close = after_eq.find("]")
            if close == -1:
                break
            after_eq = after_eq[close + 1:].strip()
        if after_eq.startswith("("):
            newick = after_eq.rstrip(";").strip()
            break

    if newick is None:
        raise ValueError("No 'tree ... = <newick>' line found in NEXUS bytes.")

    root = _parse_newick(newick)
    if translate:
        _apply_translate(root, translate)

    for node in _collect_nodes(root):
        node.is_tip = not bool(node.children)
        if node.is_tip:
            node.name = _strip_outer_quotes(node.name)

    _assign_layout(root)
    return root, translate


# ---------------------------------------------------------------------------
# Plotly trace builders
# ---------------------------------------------------------------------------

def build_tree_traces(
    root:        Node,
    x_offset:    float = 0.0,
    x_scale:     float = 1.0,
    x_flip:      bool  = False,
    highlight:   set[str] | None = None,
    label_side:  str  = "right",   # "right" | "left"
    show_labels: bool = True,
) -> list[dict]:
    """Build Plotly trace dicts for a rectangular tree layout.

    Args:
        root:       Laid-out root Node.
        x_offset:   Horizontal shift (used to place the two trees side by side).
        x_scale:    Multiply all x coordinates by this factor.
        x_flip:     Mirror the tree horizontally (for the right-hand tree).
        highlight:  Set of tip names to colour red.  Others are grey.
        label_side: Which side to draw tip labels.
        show_labels: Whether to draw tip name annotations.

    Returns:
        List of Plotly trace dicts (go.Scatter-compatible).
    """
    highlight = highlight or set()
    nodes     = _collect_nodes(root)
    tips      = [n for n in nodes if n.is_tip]

    def tx(x: float) -> float:
        scaled = x * x_scale
        return (x_offset - scaled) if x_flip else (x_offset + scaled)

    # ── Branch segments (L-shaped) ──────────────────────────────────────────
    branch_x: list[float | None] = []
    branch_y: list[float | None] = []

    for node in nodes:
        for child in node.children:
            # Horizontal arm from parent x to child x at child y.
            branch_x += [tx(node.x), tx(child.x), None]
            branch_y += [child.y,    child.y,      None]
            # Vertical arm at parent x spanning child y range.
        if node.children:
            ys = [c.y for c in node.children]
            branch_x += [tx(node.x), tx(node.x), None]
            branch_y += [min(ys),    max(ys),     None]

    traces = [dict(
        type="scatter",
        x=branch_x, y=branch_y,
        mode="lines",
        line=dict(color="#888888", width=1),
        hoverinfo="skip",
        showlegend=False,
    )]

    # ── Tip markers ─────────────────────────────────────────────────────────
    tip_x_hi, tip_y_hi, tip_names_hi = [], [], []
    tip_x_lo, tip_y_lo, tip_names_lo = [], [], []

    for tip in tips:
        if tip.name in highlight:
            tip_x_hi.append(tx(tip.x))
            tip_y_hi.append(tip.y)
            tip_names_hi.append(tip.name)
        else:
            tip_x_lo.append(tx(tip.x))
            tip_y_lo.append(tip.y)
            tip_names_lo.append(tip.name)

    if tip_x_lo:
        traces.append(dict(
            type="scatter",
            x=tip_x_lo, y=tip_y_lo,
            mode="markers",
            marker=dict(color="#aaaaaa", size=6),
            text=tip_names_lo,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))

    if tip_x_hi:
        traces.append(dict(
            type="scatter",
            x=tip_x_hi, y=tip_y_hi,
            mode="markers",
            marker=dict(color="#e63946", size=8),
            text=tip_names_hi,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))

    return traces

