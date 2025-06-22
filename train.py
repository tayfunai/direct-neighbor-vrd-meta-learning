# TODO: - Create multiple model for GNN
import argparse

import torch
from dgl import add_self_loop
from torch.nn import CrossEntropyLoss
from torch.nn.functional import relu
from torch.optim import Adam
from tqdm import tqdm
from src.dataloader.meta_graph_dataset import MetaGraphDataset  # You wrote this before
from torch.utils.data import DataLoader

from torchmetrics.functional.classification import (
    multilabel_accuracy,
)

from args import train_subparser
from src.dataloader.graph_dataset import GraphDataset
from src.graph_pack.graph_model import WGCN
from src.utils.setup_logger import logger
from src.utils.utils import compute_f1_score


def train(
    model,
    meta_loader,
    num_class,
    lr_inner=0.01,
    inner_steps=1,
    epochs=50,
    device="cuda"
):
    loss_fct = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    all_support_losses = []
    all_query_losses = []
    all_query_accuracies = []

    for epoch in range(epochs):
        model.train()

        epoch_loss = 0
        for episode in tqdm(meta_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            support_graphs = episode['support']  # List of graphs
            query_graphs = episode['query']      # List of graphs
            company = episode['company'][0]      # Company name string

            # Move graphs to device
            support_graphs = [g.to(device) for g in support_graphs]
            query_graphs = [g.to(device) for g in query_graphs]

            # Clone model weights for inner loop adaptation (can use higher library for simplicity)
            # For now, simple placeholder: do one forward/backward on support, then optimize on query

            # Inner loop on support set
            inner_optimizer = Adam(model.parameters(), lr=lr_inner)
            for _ in range(inner_steps):
                inner_loss = 0
                for g in support_graphs:
                    features = g.ndata['features'].to(torch.float64)
                    labels = g.ndata['label']
                    edge_weight = g.edata['weight'].double().to(device)

                    logits = model(g, features, edge_weight)
                    loss = loss_fct(logits.squeeze(), labels)
                    inner_loss += loss
                inner_optimizer.zero_grad()
                inner_loss.backward()
                inner_optimizer.step()

            # Outer loop on query set
            model.eval()
            outer_loss = 0
            with torch.no_grad():
                for g in query_graphs:
                    features = g.ndata['features'].to(torch.float64)
                    labels = g.ndata['label']
                    edge_weight = g.edata['weight'].double().to(device)

                    logits = model(g, features, edge_weight)
                    loss = loss_fct(logits.squeeze(), labels)
                    outer_loss += loss.item()

            epoch_loss += outer_loss
        logger.info(f"Epoch {epoch} meta-loss: {epoch_loss/len(meta_loader)}")

    # best_val_acc = 0
    # best_val_f1 = 0
    # best_test_f1 = 0

    # FIXME: Convert this in graph creattion to float32
    # features = g.ndata["features"].to(torch.float64)
    # labels = g.ndata["label"]
    #
    # train_list, val_list, test_list = [], [], []
    # loss_train, loss_val, loss_test = [], [], []
    # for e in tqdm(range(epochs)):
    #     # Forward
    #
    #     logits = model(g, features, edge_weight)
    #     f1_score_train = compute_f1_score(
    #         labels[train_mask].view(-1), logits[train_mask].view(-1)
    #     )
    #     accuracy_train = multilabel_accuracy(
    #         logits[train_mask].squeeze(dim=1),
    #         labels[train_mask].squeeze(dim=1),
    #         num_labels=num_class,
    #         average="macro",
    #     )
    #
    #     loss = loss_fct(labels[train_mask], logits[train_mask].squeeze(dim=1))
    #     loss_v = loss_fct(labels[val_mask], logits[val_mask].squeeze(dim=1))
    #     loss_t = loss_fct(labels[test_mask], logits[test_mask].squeeze(dim=1))
    #     loss_train.append(loss)
    #     loss_val.append(loss_v)
    #     loss_test.append(loss_t)
    #
    #     f1_score_val = compute_f1_score(
    #         labels[val_mask].view(-1), logits[val_mask].view(-1)
    #     )
    #     accuracy_val = multilabel_accuracy(
    #         logits[val_mask].squeeze(dim=1),
    #         labels[val_mask].squeeze(dim=1),
    #         num_labels=num_class,
    #         average="macro",
    #     )
    #
    #     f1_score_test = compute_f1_score(
    #         labels[test_mask].view(-1), logits[test_mask].view(-1)
    #     )
    #     accuracy_test = multilabel_accuracy(
    #         logits[test_mask].squeeze(dim=1),
    #         labels[test_mask].squeeze(dim=1),
    #         num_labels=num_class,
    #         average="macro",
    #     )
    #     train_list.append(f1_score_train)
    #     val_list.append(f1_score_val)
    #     test_list.append(f1_score_test)
    #     # Save the best validation accuracy and the corresponding test accuracy.
    #     if best_val_acc < accuracy_val:
    #         best_val_acc = accuracy_val
    #         # best_test_acc = test_acc
    #     if best_val_f1 < f1_score_val:
    #         best_val_f1 = f1_score_val
    #
    #     if best_test_f1 < f1_score_test:
    #         best_test_f1 = f1_score_test
    #     if best_val_acc < accuracy_test:
    #         best_val_acc = accuracy_test
    #
    #     # Backward
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     if e % 10 == 0:
    #         logger.debug(
    #             f"Epochs: {e}/{epochs}, Train F1-score: {f1_score_train}, Val F1-score: {f1_score_val}, Train Accuracy: "
    #             f"{accuracy_train}, Val Accuracy: {accuracy_val}, Best Accuracy: {best_val_acc}, Best F1-score: {best_val_f1}, Best Test F1-score: {best_test_f1}"
    #         )
    # return train_list, val_list, test_list, loss_train, loss_val, loss_test

# ======== Main script ========


if __name__ == "__main__":
    torch.manual_seed(0)
    main_parser = argparse.ArgumentParser()
    subparsers = main_parser.add_subparsers(dest="subcommand", help="Choose subcommand")
    train_subparser(subparsers)
    args = main_parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"


    data_name = args.dataset
    path = args.path
    hidden_size = args.hidden_size
    nbr_hidden_layer = args.hidden_layers
    lr = args.learning_rate
    epochs = args.epochs

    dataset = GraphDataset(data_name, path=path, split="train")
    # Create episodic meta dataset, define support and query sizes (k-shot)
    k_support = 5
    q_query = 5
    meta_dataset = MetaGraphDataset(dataset.graphs, k_support=1, q_query=1)

    meta_loader = DataLoader(meta_dataset, batch_size=1, shuffle=True)  # batch=1 episode per iteration

    # graph_train = dataset[True].to(device)
    # graph_train = add_self_loop(graph_train)
    # train_mask = dataset.train_mask
    # val_mask = dataset.val_mask
    # test_mask = dataset.test_mask
    # Initialize model
    dummy_graph = dataset.graphs[0].to(device)

    model = WGCN(
        dummy_graph.ndata["features"].shape[2],
        hidden_size,
        dataset.num_classes,
        nbr_hidden_layer,
        relu,
    ).to(device)
    # TODO: here sometime float some time double
    model.double()
    train(model, meta_loader, lr_inner=lr, inner_steps=1, epochs=epochs, device=device)

