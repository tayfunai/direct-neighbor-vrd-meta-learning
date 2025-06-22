import random
from collections import defaultdict

class MetaGraphDataset:
    def __init__(self, graphs, k_support=2, q_query=6):
        """
        Args:
            graphs (list): List of DGL graphs with attribute 'label_class' (company name).
            k_support (int): Number of support samples per episode.
            q_query (int): Number of query samples per episode.
        """
        self.k = k_support
        self.q = q_query

        # Group graphs by their label_class (company)
        self.company_to_graphs = defaultdict(list)
        for g in graphs:
            if hasattr(g, 'label_class') and g.label_class is not None:
                self.company_to_graphs[g.label_class].append(g)

        # Filter companies with enough graphs
        self.companies = [
            c for c, gs in self.company_to_graphs.items() if len(gs) >= (self.k + self.q)
        ]

    def __len__(self):
        return len(self.companies)

    def __getitem__(self, idx):
        company = self.companies[idx]
        graphs = self.company_to_graphs[company]

        # Randomly sample support and query graphs
        selected = random.sample(graphs, self.k + self.q)
        support = selected[:self.k]
        query = selected[self.k:]

        return {
            "company": company,
            "support": support,
            "query": query,
        }
