import torch as th
from torch import nn
import dgl
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax


class EOPA(nn.Module):
    def __init__(self, input_dim, output_dim, activation=None, batch_norm=True):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.gru = nn.GRU(input_dim, input_dim, batch_first=True)
        self.fc_self = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neigh = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = activation

    def reducer(self, nodes):
        m = nodes.mailbox['m']
        _, hn = self.gru(m)  # hn: (1, batch_size, d)
        return {'neigh': hn.squeeze(0)}

    def forward(self, mg, feat):
        with mg.local_scope():
            if self.batch_norm is not None:
                feat = self.batch_norm(feat)
            mg.ndata['ft'] = feat
            if mg.number_of_edges() > 0:
                mg.update_all(fn.copy_u('ft', 'm'), self.reducer)
                neigh = mg.ndata['neigh']
                rst = self.fc_self(feat) + self.fc_neigh(neigh)
            else:
                rst = self.fc_self(feat)
            if self.activation is not None:
                rst = self.activation(rst)
            return rst


class SGAT(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, activation=None, batch_norm=True
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.fc_q = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, output_dim, bias=False)
        self.attn_e = nn.Linear(hidden_dim, 1, bias=False)
        self.activation = activation

    def forward(self, sg, feat):
        with sg.local_scope():
            if self.batch_norm is not None:
                feat = self.batch_norm(feat)
            q = self.fc_q(feat)
            k = self.fc_k(feat)
            v = self.fc_v(feat)
            sg.ndata.update({'q': q, 'k': k, 'v': v})
            sg.apply_edges(fn.u_add_v('q', 'k', 'e'))
            e = self.attn_e(th.sigmoid(sg.edata['e']))
            sg.edata['a'] = edge_softmax(sg, e)
            sg.update_all(fn.u_mul_e('v', 'a', 'm'), fn.sum('m', 'ft'))
            rst = sg.ndata['ft']
            if self.activation is not None:
                rst = self.activation(rst)
            return rst


class AttnReadout(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_norm=True):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.fc_u = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, hidden_dim, bias=True)
        self.attn_e = nn.Linear(hidden_dim, 1, bias=False)
        if output_dim != input_dim:
            self.fc_out = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.fc_out = nn.Identity()

    def forward(self, g, feat, last_nodes):
        with g.local_scope():
            if self.batch_norm is not None:
                feat = self.batch_norm(feat)
            feat_u = self.fc_u(feat)
            feat_v = self.fc_v(feat[last_nodes])
            feat_v = dgl.broadcast_nodes(g, feat_v)
            g.ndata['e'] = self.attn_e(th.sigmoid(feat_u + feat_v))
            alpha = dgl.softmax_nodes(g, 'e')
            g.ndata['w'] = feat * alpha
            rst = dgl.sum_nodes(g, 'w')
            rst = self.fc_out(rst)
            return rst


class LESSR(nn.Module):
    def __init__(self, num_items, embedding_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(num_items, embedding_dim, max_norm=1)
        self.indices = nn.Parameter(
            th.arange(num_items, dtype=th.long), requires_grad=False
        )
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        input_dim = embedding_dim
        for i in range(num_layers):
            if i % 2 == 0:
                layer = EOPA(
                    input_dim,
                    embedding_dim,
                    activation=nn.PReLU(embedding_dim),
                    batch_norm=True,
                )
            else:
                layer = SGAT(
                    input_dim,
                    embedding_dim,
                    embedding_dim,
                    activation=nn.PReLU(embedding_dim),
                    batch_norm=True,
                )
            input_dim += embedding_dim
            self.layers.append(layer)
        self.readout = AttnReadout(
            input_dim, embedding_dim, embedding_dim, batch_norm=True
        )
        input_dim += embedding_dim
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.fc_sr = nn.Linear(input_dim, embedding_dim, bias=False)

    def forward(self, mg, sg):
        iid = mg.ndata['iid']
        feat = self.embedding(iid)
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                out = layer(mg, feat)
            else:
                out = layer(sg, feat)
            feat = th.cat([out, feat], dim=1)
        last_nodes = mg.filter_nodes(lambda nodes: nodes.data['last'] == 1)
        sr_g = self.readout(mg, feat, last_nodes)
        sr_l = feat[last_nodes]
        sr = th.cat([sr_l, sr_g], dim=1)
        sr = self.fc_sr(self.batch_norm(sr))
        logits = sr @ self.embedding(self.indices).t()
        return logits

