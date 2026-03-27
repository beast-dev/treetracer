"""Server-side state for data too large to send through dcc.Store.

The RF distance matrix for n trees is n×n integers — for 4000 trees that's
~80 MB of JSON. Sending this through Dash's callback response / dcc.Store
(which serializes to JSON and transfers to the browser) is too slow and can
crash browser tabs. Instead, the matrix lives here in Python memory, and
only lightweight metadata (tree names, dimensions) goes through the store.
"""

# Distance matrix: kept server-side, never serialized to JSON.
# "names" is also stored in the dcc.Store for client-side group extraction.
distmat = {
    "names": None,   # list[str] — tree identifiers
    "matrix": None,  # list[list[int]] — n×n RF distance matrix
}


def clear_distmat():
    """Reset the server-side distance matrix."""
    distmat["names"] = None
    distmat["matrix"] = None
