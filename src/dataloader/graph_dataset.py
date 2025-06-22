import os
import torch
from datasets import load_dataset
from dgl import load_graphs, batch
from dgl.data import DGLDataset
from torch import zeros
from src.utils.setup_logger import logger

def extract_company_name_from_graph(graph, label_names):
    words = graph.graph_words
    tags = graph.ndata["label"].tolist()

    company_tokens = []
    collecting = False
    for word, tag_id in zip(words, tags):
        label = label_names[tag_id]
        if label == "B-COMPANY":
            if collecting:
                break
            collecting = True
            company_tokens = [word]
        elif label == "I-COMPANY" and collecting:
            company_tokens.append(word)
        elif collecting:
            break

    return " ".join(company_tokens) if company_tokens else None



class GraphDataset(DGLDataset):
    # FIXME: CONVERT THIS TO X, Y, W, H
    def __init__(self, data_name: str, path: str = "data/", split: str = "train"):
        dataset_paths = {"FUNSD", "CORD", "SROIE", "WILDRECEIPT", "XFUND"}
        if data_name not in dataset_paths:
            raise Exception(
                f"{data_name} Invalid dataset name. Please provide a valid dataset name."
            )

        # Load the original dataset to get label names dynamically
        if data_name == "SROIE":
            raw_dataset = load_dataset("darentang/sroie")
            self.label_names = raw_dataset["train"].features["ner_tags"].feature.names
        else:
            # For other datasets, implement similarly or hardcode as fallback
            self.label_names = [...]  # your fallback label names


        self.split = split  # "train" or "test"
        self.graphs = []

        dataset_path = os.path.join(path, data_name, split)
        self.num_classes = {
            "FUNSD": 4,
            "CORD": 30,
            "SROIE": 5,
            "WILDRECEIPT": 26,
            "XFUND": 4,
        }[data_name]

        ext = "bin"

        for file in os.listdir(dataset_path):
            if file.endswith(ext):
                g = load_graphs(os.path.join(dataset_path, file))[0][0]
                g.graph_id = file
                g.label_class = extract_company_name_from_graph(g, self.label_names)  # implement this!
                self.graphs.append(g)

        # graph_list_train = [
        #     load_graphs(os.path.join(dataset_path, "train", file))[0][0]
        #     for file in os.listdir(os.path.join(dataset_path, "train"))
        #     if file.endswith(ext)
        # ]
        # graph_list_test = [
        #     load_graphs(os.path.join(dataset_path, "test", file))[0][0]
        #     for file in os.listdir(os.path.join(dataset_path, "test"))
        #     if file.endswith(ext)
        # ]

        ## Meta learning change
        # ==============================

        # self.graph_train = graph_list_train
        # self.graph_test = graph_list_test

        # ==============================

        # self.graph_train = batch(graph_list_train)
        # self.graph_test = batch(graph_list_test)
        #
        # self.graph = batch([self.graph_train, self.graph_test])
        super().__init__(name="GraphDataset")

    """
    def process(self):
        n_nodes = self.graph.number_of_nodes()
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        train_mask[:n_train] = True
        val_mask[n_train : n_train + n_val] = True
        test_mask[n_train + n_val :] = True

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
    """

    # def __getitem__(self, train: bool):
    #     return self.graph if train else self.graph_test
    #
    # def __len__(self):
    #     return 1

    def __len__(self):
        return len(self.graphs)  # or .graphs_test depending on split

    def __getitem__(self, idx):
        return self.graphs[idx]  # or self.graphs_test[idx]
