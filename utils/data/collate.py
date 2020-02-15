from collections import Counter
import numpy as np
import torch as th
import dgl


def label_last(g, last_nid):
    is_last = np.zeros(g.number_of_nodes(), dtype=np.int32)
    is_last[last_nid] = 1
    g.ndata['last'] = is_last
    return g


def seq_to_eop_multigraph(seq):
    items = np.unique(seq)
    iid2nid = {iid: i for i, iid in enumerate(items)}
    num_nodes = len(items)

    g = dgl.DGLGraph(multigraph=True)
    g.add_nodes(num_nodes)
    g.ndata['iid'] = items

    if len(seq) > 1:
        seq_nid = [iid2nid[iid] for iid in seq]
        src = seq_nid[:-1]
        dst = seq_nid[1:]
        # edges are added in the order of their occurrences.
        g.add_edges(src, dst)

    label_last(g, iid2nid[seq[-1]])
    return g


def seq_to_shortcut_graph(seq):
    items = np.unique(seq)
    iid2nid = {iid: i for i, iid in enumerate(items)}
    num_nodes = len(items)

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.ndata['iid'] = items

    seq_nid = [iid2nid[iid] for iid in seq]
    counter = Counter(
        [(seq_nid[i], seq_nid[j]) for i in range(len(seq)) for j in range(i, len(seq))]
    )
    edges = counter.keys()
    src, dst = zip(*edges)
    # edges are added in the order of their first occurrences.
    g.add_edges(src, dst)

    return g


def collate_fn(samples):
    seqs, labels = zip(*samples)
    inputs = []
    for seq_to_graph in [seq_to_eop_multigraph, seq_to_shortcut_graph]:
        graphs = list(map(seq_to_graph, seqs))
        bg = dgl.batch(graphs)
        inputs.append(bg)
    labels = th.tensor(labels, dtype=th.long)
    return inputs, labels
