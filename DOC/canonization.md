# WL-based graph canonization

## Provenance

This implementation is adapted from the WL canonical-ordering code in
[GraViti](https://github.com/RomanBresson/GraViti), by Roman Bresson.  The
relevant source supplied with this repository is `old/canonization.py`.

The COLORING implementation retains only GraViti's `ordering="wl"` idea.  It
does not use the geometry ordering and does not assume node positions are
available.

## Input and output

The public function uses the repository graph convention:

```text
graph = (adjacency_matrix, node_colors)
canonized_graph = canonize_graph(graph, wl_iterations=3)
```

The output is another `(adjacency_matrix, node_colors)` tuple.  It represents
exactly the same colored graph with a deterministic node permutation applied:

```text
canonical_node_colors = node_colors[permutation]
canonical_adjacency = adjacency_matrix[permutation, permutation]
```

The input must be a simple undirected graph with a square, symmetric, binary
adjacency matrix, no self-loops, and one integer color per node.

## Review of the proposed GraViti implementation

The proposed implementation is concise and well suited to moderate COLORING
graphs.  Its useful ingredients are:

- labeled 1-WL refinement to summarize local colored structure;
- one structurally selected root;
- shortest-path distances from that root;
- a second, distant anchor to reduce remaining symmetries;
- degree and original index as final deterministic tie-breakers.

The original GraViti code also uses `(y, x)` node positions to break WL ties.
Those positions are unavailable in the requested API, so the position keys are
removed.  The remaining original-index tie-break is deterministic for a fixed
input array.

This distinction matters: the method is a practical **canonical ordering
heuristic**, not a complete mathematical graph canonizer.  If WL, degrees, and
the two distance vectors leave several structurally symmetric nodes tied, the
final order depends on their input indices.  Thus two isomorphic graphs can in
rare symmetric cases produce different serialized outputs.  Achieving a
guaranteed canonical form would require an exact individualization/refinement
search or a specialized planar canonization algorithm, which is outside the
GraViti `ordering="wl"` implementation requested here.

## Pseudocode: WL refinement

```text
WL_LABELS(adjacency_matrix, node_colors, max_iterations):
    labels = integer copy of node_colors
    neighbors[v] = indices adjacent to v, for every node v

    repeat at most max(1, max_iterations) times:
        signatures = empty list

        for each node v:
            neighbor_labels = sorted(labels[u] for u in neighbors[v])
            signatures[v] = (labels[v], tuple(neighbor_labels))

        vocabulary = distinct signatures sorted lexicographically
        map each signature in vocabulary to a dense integer ID
        new_labels[v] = ID of signatures[v]

        # Refinement includes the previous label and therefore never merges
        # color classes. Equal class counts mean the partition has stabilized.
        if number_of_unique(new_labels) == number_of_unique(labels):
            labels = new_labels
            stop

        labels = new_labels

    return labels
```

## Pseudocode: BFS distances

```text
BFS_DISTANCES(adjacency_matrix, source):
    N = number of nodes
    distance = array filled with N + 1  # unreachable sentinel
    distance[source] = 0
    queue = [source]

    while queue is not empty:
        node = pop queue head
        for each neighbor not yet visited:
            distance[neighbor] = distance[node] + 1
            append neighbor to queue

    return distance
```

## Pseudocode: position-free WL ordering

```text
WL_CANONICAL_PERMUTATION(adjacency_matrix, node_colors, wl_iterations):
    N = number of nodes
    if N <= 1:
        return [0, ..., N - 1]

    degree[v] = number of neighbors of v
    wl_label = WL_LABELS(adjacency_matrix, node_colors, wl_iterations)

    root = node minimizing:
        (wl_label[node], degree[node], original_index[node])

    distance_from_root = BFS_DISTANCES(adjacency_matrix, root)

    anchor = node maximizing:
        (distance_from_root[node],
         degree[node],
         -wl_label[node],
         original_index[node])

    distance_from_anchor = BFS_DISTANCES(adjacency_matrix, anchor)

    permutation = nodes sorted by:
        (wl_label[node],
         distance_from_root[node],
         distance_from_anchor[node],
         degree[node],
         original_index[node])

    return permutation
```

## Pseudocode: public function

```text
CANONIZE_GRAPH(graph, wl_iterations=3):
    adjacency_matrix, node_colors = validate and normalize graph
    permutation = WL_CANONICAL_PERMUTATION(
        adjacency_matrix, node_colors, wl_iterations
    )

    canonized_adjacency = adjacency_matrix indexed by permutation on both axes
    canonized_colors = node_colors indexed by permutation
    return (canonized_adjacency, canonized_colors)
```

The operation preserves node colors, edges, planarity, and graph isomorphism;
only node indices change.
