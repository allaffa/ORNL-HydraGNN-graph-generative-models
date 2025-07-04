import torch
import torch_geometric
import os

import rdkit.Chem as Chem

from torch_geometric.transforms import BaseTransform

class FullyConnectGraph(BaseTransform):
    def __call__(self, data):
        num_nodes = data.x.shape[0]
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).T  # Get all unique pairs
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Make bidirectional
        data.edge_index = edge_index  # Modify in place
        return data  # Return the modified data object

# def fully_connect_graph(data):
#     num_nodes = data.x.shape[0]
#     edge_index = torch.combinations(
#         torch.arange(num_nodes), r=2
#     ).T  # Get all unique pairs
#     edge_index = torch.cat(
#         [edge_index, edge_index.flip(0)], dim=1
#     )  # Make bidirectional
#     return edge_index


# def make_fully_connected(dataset):
#     for i,data in enumerate(dataset):
#         edge_index = fully_connect_graph(data)
#         dataset[i].edge_index = edge_index
#     print(dataset[0].edge_index)
#     return dataset


def write_pdb_file(data, output_file):
    """
    Create a PDB file from the output of the denoising model.
    Creates a RDKit molecule and writes it to a PDB file to make
    loading files in RDKit easier when analyzing.

    Args:
    -----
    data (Data):
        The output of the denoising model.
    output_file (str):
        The path to the output PDB file.
    """
    atom_map = [1, 6, 7, 8, 9]  # HCNOF
    atom_type = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
    # Get atom types
    x_argmax = torch.argmax(data.x, dim=1).cpu().numpy()
    atoms = [atom_type[atom_map[t]] for t in x_argmax]
    # Get positions
    pos = data.pos.cpu().numpy().tolist()

    # Create Empty Mol
    mol = Chem.RWMol()

    # Add atoms to mol
    for atom in atoms:
        mol.AddAtom(Chem.Atom(atom))

    # Add a new conformer. The argument 0 means that it will have 0 atoms, but atoms will be added later.
    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))

    # Add atom positions
    for i, p in enumerate(pos):
        mol.GetConformer().SetAtomPosition(i, p)

    # Save molecule to PDB file
    Chem.MolToPDBFile(mol, output_file)


def get_marg_dist(root_path: str) -> torch.Tensor:
    """
    Returns the marginal distribution of atom types in the QM9 dataset.
    Used for the noising model for discrete node features.

    Args:
    -----
    root_path (str):
        The path to the (stored) QM9 dataset.
    """
    # Load the QM9 dataset
    qm9 = torch_geometric.datasets.QM9(root=os.path.join(root_path))

    # Find the marginal distribution of atom types in the dataset
    # Sum the atom types to get the marginal distribution
    total_atom_types = qm9.x[:, :5].sum(dim=0)
    # Normalize the distribution and return
    return total_atom_types / total_atom_types.sum()
