import time

import torch as th
from torch import nn, optim


# ignore weight decay for bias and batch norm
def fix_weight_decay(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'batch_norm' in name:
            no_decay.append(param)
        else:
            decay.append(param)
    params = [{'params': decay}, {'params': no_decay, 'weight_decay': 0}]
    return params


def prepare_batch(batch, device):
    inputs, labels = batch
    inputs_gpu = [x.to(device) for x in inputs]
    labels_gpu = labels.to(device)
    return inputs_gpu, labels_gpu


def evaluate(model, data_loader, device, cutoff=20):
    model.eval()
    mrr = th.tensor(0.0)
    hit = th.tensor(0.0)
    num_samples = 0
    with th.no_grad():
        for batch in data_loader:
            inputs, labels = prepare_batch(batch, device)
            logits = model(*inputs)
            batch_size = logits.size(0)
            num_samples += batch_size
            _, topk = logits.topk(k=cutoff)
            labels = labels.unsqueeze(-1)
            hit_ranks = th.where(topk == labels)[1] + 1
            r_ranks = 1 / hit_ranks.to(th.float32)
            hit += hit_ranks.numel()
            mrr += r_ranks.sum()
    return mrr / num_samples, hit / num_samples


class TrainRunner:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        device,
        lr=1e-3,
        weight_decay=0,
        patience=3,
    ):
        self.model = model
        if weight_decay > 0:
            params = fix_weight_decay(model)
            self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epoch = 0
        self.batch = 0
        self.patience = patience

    def train(self, epochs, log_interval=100):
        max_mrr = 0
        max_hit = 0
        bad_counter = 0
        t = time.time()
        mean_loss = 0
        for epoch in range(epochs):
            self.model.train()
            for batch in self.train_loader:
                inputs, labels = prepare_batch(batch, self.device)
                self.optimizer.zero_grad()
                logits = self.model(*inputs)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                mean_loss += loss / log_interval
                if self.batch > 0 and self.batch % log_interval == 0:
                    print(
                        f'Batch {self.batch}: Loss = {mean_loss.item():.4f}, Time Elapsed = {time.time() - t:.2f}s'
                    )
                    t = time.time()
                    mean_loss = 0
                self.batch += 1

            mrr, hit = evaluate(self.model, self.test_loader, self.device)

            print(f'Epoch {self.epoch}: MRR = {mrr * 100:.3f}%, Hit = {hit * 100:.3f}%')

            if mrr < max_mrr and hit < max_hit:
                bad_counter += 1
                if bad_counter == self.patience:
                    break
            else:
                bad_counter = 0
            max_mrr = max(max_mrr, mrr)
            max_hit = max(max_hit, hit)
            self.epoch += 1

