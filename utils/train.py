import time
from collections import defaultdict

import torch as th
from torch import nn, optim


# ignore weight decay for parameters in bias, batch norm and activation
def fix_weight_decay(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(map(lambda x: x in name, ['bias', 'batch_norm', 'activation'])):
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


def evaluate(model, data_loader, device, Ks=[20]):
    model.eval()
    num_samples = 0
    max_K = max(Ks)
    results = defaultdict(float)
    with th.no_grad():
        for batch in data_loader:
            inputs, labels = prepare_batch(batch, device)
            logits = model(*inputs)
            batch_size = logits.size(0)
            num_samples += batch_size
            topk = th.topk(logits, k=max_K, sorted=True)[1]
            labels = labels.unsqueeze(-1)
            for K in Ks:
                hit_ranks = th.where(topk[:, :K] == labels)[1] + 1
                hit_ranks = hit_ranks.float().cpu()
                results[f'HR@{K}'] += hit_ranks.numel()
                results[f'MRR@{K}'] += hit_ranks.reciprocal().sum().item()
                results[f'NDCG@{K}'] += th.log2(1 + hit_ranks).reciprocal().sum().item()
    for metric in results:
        results[metric] /= num_samples
    return results


def print_results(results, epochs=None):
    print('Metric\t' + '\t'.join(results.keys()))
    print(
        'Value\t' +
        '\t'.join([f'{round(val * 100, 2):.2f}' for val in results.values()])
    )
    if epochs is not None:
        print('Epoch\t' + '\t'.join([str(epochs[metric]) for metric in results]))


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
        Ks=[20],
    ):
        self.model = model
        if weight_decay > 0:
            params = fix_weight_decay(model)
        else:
            params = model.parameters()
        self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epoch = 0
        self.batch = 0
        self.patience = patience
        self.Ks = Ks

    def train(self, epochs, log_interval=100):
        max_results = defaultdict(float)
        max_epochs = defaultdict(int)
        bad_counter = 0
        t = time.time()
        mean_loss = 0
        for epoch in range(epochs):
            self.model.train()
            for batch in self.train_loader:
                inputs, labels = prepare_batch(batch, self.device)
                self.optimizer.zero_grad()
                logits = self.model(*inputs)
                loss = nn.functional.cross_entropy(logits, labels)
                loss.backward()
                self.optimizer.step()
                mean_loss += loss.item() / log_interval
                if self.batch > 0 and self.batch % log_interval == 0:
                    print(
                        f'Batch {self.batch}: Loss = {mean_loss:.4f}, Time Elapsed = {time.time() - t:.2f}s'
                    )
                    t = time.time()
                    mean_loss = 0
                self.batch += 1

            curr_results = evaluate(
                self.model, self.test_loader, self.device, Ks=self.Ks
            )

            print(f'\nEpoch {self.epoch}:')
            print_results(curr_results)

            any_better_result = False
            for metric in curr_results:
                if curr_results[metric] > max_results[metric]:
                    max_results[metric] = curr_results[metric]
                    max_epochs[metric] = self.epoch
                    any_better_result = True

            if any_better_result:
                bad_counter = 0
            else:
                bad_counter += 1
                if bad_counter == self.patience:
                    break

            self.epoch += 1
        print('\nBest results')
        print_results(max_results, max_epochs)
        return max_results
