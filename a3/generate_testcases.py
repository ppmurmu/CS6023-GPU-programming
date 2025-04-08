import os
import random

def prepare(edges, a, b, folder_path, filename):
    os.makedirs(folder_path, exist_ok=True)  # Ensure the folder exists
    file_path = os.path.join(folder_path, filename)  # Create full file path
    
    with open(file_path, "w") as f:
        f.write(f"{a} {b}\n")  # Write first line
        for edge in edges:
            f.write(" ".join(map(str, edge)) + "\n")  # Write edges line by line

n = 1

limit = 1000000
options = ["green", "traffic", "normal", "dept"]

for i in range(n):
    edges = []
    v=limit
    # v = random.randint(3, limit)
    print(v)

    # Set to store unique edges (undirected: store as (min, max))
    edge_set = set()

    # Step 1: Generate a spanning tree to ensure connectivity
    nodes = list(range(v))
    random.shuffle(nodes)  # Shuffle to get a random tree structure
    for j in range(1, v):  # Connect each node to a previous one
        parent = random.randint(0, j - 1)
        u, w = nodes[parent], nodes[j]
        edge_set.add((min(u, w), max(u, w)))  # Store in sorted order
        edges.append([u, w, random.randint(1, 1000), random.choice(options)])

    # Step 2: Add extra random edges (not forming a complete graph)
    extra_edges = random.randint(v, v * 2)  # Add random edges (but not too many)
    while len(edge_set) < extra_edges:
        u, w = random.sample(nodes, 2)  # Pick two different nodes
        if (min(u, w), max(u, w)) not in edge_set:
            edge_set.add((min(u, w), max(u, w)))
            edges.append([u, w, random.randint(1, 1000), random.choice(options)])

    prepare(edges, v, len(edges), r"./", f"test{i+1}.txt")
