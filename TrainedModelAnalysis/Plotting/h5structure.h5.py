import h5py
import matplotlib.pyplot as plt

def build_tree(name, node, prefix='', is_last=True):
    """
    Recursively build a list of strings representing the HDF5 tree,
    with ASCII connectors.
    """
    lines = []
    connector = '└── ' if is_last else '├── '
    lines.append(prefix + connector + name + ('/' if isinstance(node, h5py.Group) else ''))
    if isinstance(node, h5py.Group):
        children = list(node.items())
        for idx, (child_name, child_node) in enumerate(children):
            last = idx == len(children) - 1
            child_prefix = prefix + ('    ' if is_last else '│   ')
            lines.extend(build_tree(child_name, child_node, child_prefix, last))
    return lines

# Build the tree lines
with h5py.File("AttentionTitrations.h5", "r") as f:
    lines = ['.']  # represent the root
    items = list(f.items())
    for idx, (name, node) in enumerate(items):
        last = (idx == len(items) - 1)
        lines.extend(build_tree(name, node, '', last))

# Render to PNG
fig = plt.figure(figsize=(8, 10))
plt.text(0, 1, "\n".join(lines), va='top', family='monospace')
plt.axis('off')
plt.savefig("AttentionTitrations_tree.png", dpi=300, bbox_inches='tight')
plt.close()
