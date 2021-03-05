import torch as th
from torch import nn
import dgl
import dgl.ops as F
import dgl.function as fn


class EOPA(nn.Module):
    def __init__(
        self, input_dim, output_dim, batch_norm=True, feat_drop=0.0, activation=None
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.gru = nn.GRU(input_dim, input_dim, batch_first=True)
        self.fc_self = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neigh = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = activation

    def reducer(self, nodes):
        m = nodes.mailbox['m']  # (num_nodes, deg, d)
        # m[i]: the messages passed to the i-th node with in-degree equal to 'deg'
        # the order of messages follows the order of incoming edges
        # since the edges are sorted by occurrence time when the EOP multigraph is built
        # the messages are in the order required by EOPA
        _, hn = self.gru(m)  # hn: (1, num_nodes, d)
        return {'neigh': hn.squeeze(0)}

    def forward(self, mg, feat):
        with mg.local_scope():
            if self.batch_norm is not None:
                feat = self.batch_norm(feat)
            mg.ndata['ft'] = self.feat_drop(feat)
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
        self,
        input_dim,
        hidden_dim,
        output_dim,
        batch_norm=True,
        feat_drop=0.0,
        activation=None,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_q = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_e = nn.Linear(hidden_dim, 1, bias=False)
        self.activation = activation

    def forward(self, sg, feat):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        feat = self.feat_drop(feat)
        q = self.fc_q(feat)
        k = self.fc_k(feat)
        v = self.fc_v(feat)
        e = F.u_add_v(sg, q, k)
        e = self.fc_e(th.sigmoid(e))
        a = F.edge_softmax(sg, e)
        rst = F.u_mul_e_sum(sg, v, a)
        if self.activation is not None:
            rst = self.activation(rst)
        return rst


class AttnReadout(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        batch_norm=True,
        feat_drop=0.0,
        activation=None,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_u = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_e = nn.Linear(hidden_dim, 1, bias=False)
        self.fc_out = (
            nn.Linear(input_dim, output_dim, bias=False)
            if output_dim != input_dim else None
        )
        self.activation = activation

    def forward(self, g, feat, last_nodes):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        feat = self.feat_drop(feat)
        feat_u = self.fc_u(feat)
        feat_v = self.fc_v(feat[last_nodes])
        feat_v = dgl.broadcast_nodes(g, feat_v)
        e = self.fc_e(th.sigmoid(feat_u + feat_v))
        alpha = F.segment.segment_softmax(g.batch_num_nodes(), e)
        feat_norm = feat * alpha
        rst = F.segment.segment_reduce(g.batch_num_nodes(), feat_norm, 'sum')
        if self.fc_out is not None:
            rst = self.fc_out(rst)
        if self.activation is not None:
            rst = self.activation(rst)
        return rst


class LESSR(nn.Module):
    def __init__(
        self, num_items, embedding_dim, num_layers, batch_norm=True, feat_drop=0.0
    ):
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
                    batch_norm=batch_norm,
                    feat_drop=feat_drop,
                    activation=nn.PReLU(embedding_dim),
                )
            else:
                layer = SGAT(
                    input_dim,
                    embedding_dim,
                    embedding_dim,
                    batch_norm=batch_norm,
                    feat_drop=feat_drop,
                    activation=nn.PReLU(embedding_dim),
                )
            input_dim += embedding_dim
            self.layers.append(layer)
        self.readout = AttnReadout(
            input_dim,
            embedding_dim,
            embedding_dim,
            batch_norm=batch_norm,
            feat_drop=feat_drop,
            activation=nn.PReLU(embedding_dim),
        )
        input_dim += embedding_dim
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_sr = nn.Linear(input_dim, embedding_dim, bias=False)

    def forward(self, mg, sg=None):
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
        if self.batch_norm is not None:
            sr = self.batch_norm(sr)
        sr = self.fc_sr(self.feat_drop(sr))
        logits = sr @ self.embedding(self.indices).t()
        return logits
