# @title Peptide

from rdkit import Chem

from hypergraph_embedding import GraphEmbedding

class Peptide:
    """Represents a peptide molecule with graph embedding capabilities."""

    def __init__(self, smiles: str, num_residues: int, num_residues_in_main_loop: int):
        """Initializes a Peptide object.

        Args:
            smiles: SMILES string representation of the peptide.
            num_residues: Total number of residues in the peptide.
            num_residues_in_main_loop: Number of residues in the main loop.
        """
        self.smiles = smiles
        self.num_residues = num_residues
        self.num_residues_in_main_loop = num_residues_in_main_loop
        self.num_residues_in_branch = self.num_residues - self.num_residues_in_main_loop
        self.mol = self._prepare_molecule(smiles)
        self.graph_embedding = self._embed()

    @staticmethod
    def _prepare_molecule(smiles: str) -> Chem.Mol:
        """Prepares a RDKit molecule from SMILES string.

        Args:
            smiles: SMILES string representation of the molecule.

        Returns:
            A prepared RDKit molecule.

        Raises:
            ValueError: If molecule creation fails.
        """
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            raise ValueError(f"Failed to create molecule from SMILES: {smiles}")
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        return mol

    def _embed(self) -> 'GraphEmbedding':
        """Creates a GraphEmbedding object for the peptide."""
        return GraphEmbedding(self)

    def draw_graphs(self) -> None:
        """Draws the graph representations of the peptide."""
        self.graph_embedding.draw_graphs()
