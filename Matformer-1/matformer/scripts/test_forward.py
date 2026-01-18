import torch
from torch_geometric.data import Data
from matformer.models.pyg_att import Matformer, MatformerConfig

def make_graph(n=4, e=6):
    x = torch.randn(n, 92)
    edge_index = torch.randint(0, n, (2, e))
    edge_attr = torch.randn(e, 3)
    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    # Set batch vector for single-graph case
    g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
    # Line graph placeholder (not used in forward when angle_lattice=False)
    lg = Data(x=x.clone(), edge_index=edge_index.clone(), edge_attr=edge_attr.clone())
    lg.batch = torch.zeros(lg.x.size(0), dtype=torch.long)
    # Lattice placeholder tensor; model ignores when angle_lattice=False
    lattice = torch.randn(1, 3, 3)
    return g, lg, lattice


def test_link_log():
    cfg = MatformerConfig(name="matformer", output_features=200, link="log")
    net = Matformer(cfg)
    g, lg, lattice = make_graph()
    out = net([g, lg, lattice])
    assert out.min().item() >= 0, "Output has negative values under log link"
    print("Forward with link=log passed; output shape:", out.shape)


def test_link_identity():
    cfg = MatformerConfig(name="matformer", output_features=200, link="identity")
    net = Matformer(cfg)
    g, lg, lattice = make_graph()
    out = net([g, lg, lattice])
    print("Forward with link=identity passed; output shape:", out.shape)


if __name__ == "__main__":
    test_link_log()
    test_link_identity()