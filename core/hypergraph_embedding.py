# @title Graph Embedding

import matplotlib.collections as mc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, rdMolDescriptors, Descriptors, Draw
import torch
from torch_geometric.data import Data
from typing import Dict, List, Set, Tuple

from peptide import Peptide

class GraphEmbedding:
    """Handles graph embedding for a peptide molecule with advanced featurization."""

    def __init__(self, peptide: Peptide):
        """Initializes a GraphEmbedding object.

        Args:
            peptide: The Peptide object to create embeddings for.
        """
        self.peptide = peptide
        self.mol = peptide.mol
        self.atom_graph = self._build_atom_graph()
        self.residue_graph = self._build_residue_graph()
        self.global_node = self._build_global_node()

    def _build_atom_graph(self) -> nx.Graph:
        """Builds a graph representation of atoms in the molecule with advanced features."""
        graph = nx.Graph()

        # Compute Gasteiger charges
        AllChem.ComputeGasteigerCharges(self.mol)

        for atom in self.mol.GetAtoms():
            features = self._get_atom_features(atom)
            graph.add_node(atom.GetIdx(), **features)

        for bond in self.mol.GetBonds():
            features = self._get_bond_features(bond)
            graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), **features)

        return graph

    def _get_atom_features(self, atom: Chem.Atom) -> dict:
        """Computes comprehensive features for an atom."""
        atom_features = {
            # Basic properties
            'atomic_num': atom.GetAtomicNum(),
            'degree': atom.GetDegree(),
            'total_degree': atom.GetTotalDegree(),
            'explicit_valence': atom.GetExplicitValence(),
            'implicit_valence': atom.GetImplicitValence(),
            'formal_charge': atom.GetFormalCharge(),
            'hybridization': int(atom.GetHybridization()),
            'total_num_Hs': atom.GetTotalNumHs(),
            'num_explicit_Hs': atom.GetNumExplicitHs(),
            'num_implicit_Hs': atom.GetNumImplicitHs(),
            'atomic_mass': atom.GetMass(),
            'isotope': atom.GetIsotope(),

            # Electronic properties
            'gasteiger_charge': atom.GetDoubleProp('_GasteigerCharge'),
            'default_valency': Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum()),
            'num_radical_electrons': atom.GetNumRadicalElectrons(),

            # Topological properties
            'is_aromatic': int(atom.GetIsAromatic()),
            'is_in_ring': int(atom.IsInRing()),
            'is_in_ring_size_3': int(atom.IsInRingSize(3)),
            'is_in_ring_size_4': int(atom.IsInRingSize(4)),
            'is_in_ring_size_5': int(atom.IsInRingSize(5)),
            'is_in_ring_size_6': int(atom.IsInRingSize(6)),
            'is_in_ring_size_7': int(atom.IsInRingSize(7)),
            'num_rings': len(atom.GetOwningMol().GetRingInfo().AtomRings()),

            # Stereochemistry
            'chiral_tag': int(atom.GetChiralTag()),
            'chirality_type': str(atom.GetChiralTag()),
            'has_chiral_tag': int(atom.HasProp('_CIPCode')),

            # Contributions to molecular properties
            'crippen_contrib_logp': Crippen.MolLogP(self.mol) / self.mol.GetNumAtoms(),
            'crippen_contrib_mr': Crippen.MolMR(self.mol) / self.mol.GetNumAtoms(),
            'tpsa_contrib': rdMolDescriptors.CalcTPSA(self.mol) / self.mol.GetNumAtoms(),
            'labute_asa': float(rdMolDescriptors.CalcLabuteASA(self.mol)) / self.mol.GetNumAtoms(),

            # Environment features
            'num_neighbors': len([x for x in atom.GetNeighbors()]),
            'num_aromatic_bonds': len([b for b in atom.GetBonds() if b.GetIsAromatic()]),
            'num_single_bonds': len([b for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.SINGLE]),
            'num_double_bonds': len([b for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE]),
            'num_triple_bonds': len([b for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.TRIPLE]),

            # Protein-specific features
            'is_backbone': int(atom.GetSymbol() in ['N', 'C', 'O'] and self._is_backbone_atom(atom)),
            'is_sidechain': int(not self._is_backbone_atom(atom)),
            'is_terminal': int(atom.GetDegree() == 1)
        }

        return atom_features

    def _get_bond_features(self, bond: Chem.Bond) -> dict:
        """Computes comprehensive features for a bond."""
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()

        return {
            # Basic properties
            'bond_type': int(bond.GetBondType()),
            'bond_type_str': str(bond.GetBondType()),
            'is_conjugated': int(bond.GetIsConjugated()),
            'is_in_ring': int(bond.IsInRing()),
            'is_aromatic': int(bond.GetIsAromatic()),

            # Stereochemistry
            'stereo': int(bond.GetStereo()),
            'stereo_str': str(bond.GetStereo()),
            'is_stereochem': int(bond.GetStereo() != Chem.BondStereo.STEREONONE),

            # Rotatable bond
            'is_rotatable': int(self._is_rotatable_bond(bond)),

            # Topology
            'ring_size': min([len(ring) for ring in bond.GetOwningMol().GetRingInfo().BondRings()
                            if bond.GetIdx() in ring], default=0),

            # Bond environment
            'valence_contribution': bond.GetValenceContrib(begin_atom) + bond.GetValenceContrib(end_atom),
            'is_conjugated_to_aromatic': int(any(nb.GetIsAromatic() for nb in begin_atom.GetNeighbors()) or
                                        any(nb.GetIsAromatic() for nb in end_atom.GetNeighbors())),

            # Peptide-specific features
            'is_amide': int(self._is_secondary_amide(bond)),
            'is_peptoid': int(self._is_peptoid_bond(bond)),
            'is_disulfide': int(self._is_disulfide_bond(bond)),
            'is_backbone': int(self._is_secondary_amide(bond) or self._is_peptoid_bond(bond)),

            # Atom context
            'connects_aromatic': int(begin_atom.GetIsAromatic() and end_atom.GetIsAromatic()),
            'connects_ring': int(begin_atom.IsInRing() and end_atom.IsInRing()),
            'connects_different_rings': int(begin_atom.IsInRing() and end_atom.IsInRing() and
                                        not bond.IsInRing())
        }

    @staticmethod
    def _is_rotatable_bond(bond: Chem.Bond) -> bool:
        """Determines if a bond is rotatable."""
        return (not bond.IsInRing() and
                bond.GetBondType() == Chem.BondType.SINGLE and
                not bond.GetBeginAtom().GetAtomicNum() == 1 and
                not bond.GetEndAtom().GetAtomicNum() == 1)

    def _build_residue_graph(self) -> nx.Graph:
        """Builds a graph representation of residues in the molecule."""
        graph = nx.Graph()
        residue_bonds = self._get_residue_bonds()
        residues = self._get_residues(residue_bonds)

        for i, residue_atoms in enumerate(residues):
            backbone_atoms = self._get_backbone_atoms(residue_atoms)
            graph.add_node(f"R{i}", atoms=residue_atoms, backbone_atoms=backbone_atoms)

        for i, residue1 in enumerate(residues):
            for j, residue2 in enumerate(residues):
                if i != j and self._are_residues_connected(residue1, residue2):
                    graph.add_edge(f"R{i}", f"R{j}")

        return graph

    def _build_global_node(self) -> nx.Graph:
        """Builds a graph with a global node connected to all residues."""
        graph = nx.Graph()
        graph.add_node("Global")
        for node in self.residue_graph.nodes():
            graph.add_edge("Global", node)
        return graph

    def _get_backbone_atoms(self, residue_atoms: Set[int]) -> Set[int]:
        """Identifies backbone atoms in a residue."""
        backbone_atoms = set()
        for atom_idx in residue_atoms:
            atom = self.mol.GetAtomWithIdx(atom_idx)
            if atom.GetSymbol() in ['N', 'C', 'O']:
                if self._is_backbone_atom(atom):
                    backbone_atoms.add(atom_idx)
        return backbone_atoms

    @staticmethod
    def _is_backbone_atom(atom: Chem.Atom) -> bool:
        """Checks if an atom is part of the peptide backbone."""
        if atom.GetSymbol() == 'N':
            return any(n.GetSymbol() == 'C' for n in atom.GetNeighbors())
        elif atom.GetSymbol() == 'C':
            return any(n.GetSymbol() == 'N' for n in atom.GetNeighbors())
        elif atom.GetSymbol() == 'O':
            return any(n.GetSymbol() == 'C' for n in atom.GetNeighbors())
        return False

    def _get_residue_bonds(self) -> List[Chem.Bond]:
        """Identifies bonds that form residue boundaries."""
        return [bond for bond in self.mol.GetBonds() if self._is_residue_boundary(bond)]

    def _is_residue_boundary(self, bond: Chem.Bond) -> bool:
        """Checks if a bond forms a residue boundary."""
        return (self._is_secondary_amide(bond) or
                self._is_peptoid_bond(bond) or
                self._is_disulfide_bond(bond))

    def _is_secondary_amide(self, bond: Chem.Bond) -> bool:
        """Checks if a bond is a secondary amide bond."""
        begin_atom, end_atom = bond.GetBeginAtom(), bond.GetEndAtom()
        return (begin_atom.GetSymbol() == 'C' and end_atom.GetSymbol() == 'N' and
                self._is_carbonyl_carbon(begin_atom) and end_atom.GetTotalDegree() == 3)

    def _is_peptoid_bond(self, bond: Chem.Bond) -> bool:
        """Checks if a bond is a peptoid bond."""
        begin_atom, end_atom = bond.GetBeginAtom(), bond.GetEndAtom()
        return (begin_atom.GetSymbol() == 'C' and end_atom.GetSymbol() == 'N' and
                self._is_carbonyl_carbon(begin_atom) and end_atom.GetTotalDegree() == 3 and
                any(neighbor.GetSymbol() == 'C' for neighbor in end_atom.GetNeighbors() if neighbor != begin_atom))

    @staticmethod
    def _is_disulfide_bond(bond: Chem.Bond) -> bool:
        """Checks if a bond is a disulfide bond."""
        return (bond.GetBeginAtom().GetSymbol() == 'S' and
                bond.GetEndAtom().GetSymbol() == 'S' and
                bond.GetBondType() == Chem.BondType.SINGLE)

    def _get_residues(self, residue_bonds: List[Chem.Bond]) -> List[Set[int]]:
        """Identifies residues in the molecule."""
        all_residues = []
        visited_atoms = set()
        bond_atoms = set([atom.GetIdx() for bond in residue_bonds for atom in (bond.GetBeginAtom(), bond.GetEndAtom())])

        main_loop_residues = self._identify_residues(self.peptide.num_residues_in_main_loop, visited_atoms, bond_atoms)
        all_residues.extend(main_loop_residues)

        main_branch_residues = self._identify_residues(self.peptide.num_residues_in_branch, visited_atoms, bond_atoms)
        all_residues.extend(main_branch_residues)

        remaining_atoms = set(range(self.mol.GetNumAtoms())) - visited_atoms
        self._merge_remaining_atoms(remaining_atoms, all_residues)

        return all_residues

    def _identify_residues(self, num_residues: int, visited_atoms: Set[int], bond_atoms: Set[int]) -> List[Set[int]]:
        """Identifies a specified number of residues."""
        residues = []
        for _ in range(num_residues):
            start_atom = self._find_next_start_atom(visited_atoms, bond_atoms)
            if start_atom is None:
                break
            residue_atoms = self._get_residue_atoms(start_atom, visited_atoms, bond_atoms)
            residues.append(residue_atoms)
            visited_atoms.update(residue_atoms)
        return residues

    def _find_next_start_atom(self, visited_atoms: Set[int], bond_atoms: Set[int]) -> Chem.Atom:
        """Finds the next starting atom for residue identification."""
        for atom_idx in bond_atoms:
            if atom_idx not in visited_atoms:
                return self.mol.GetAtomWithIdx(atom_idx)
        return None

    def _get_residue_atoms(self, start_atom: Chem.Atom, visited_atoms: Set[int], bond_atoms: Set[int]) -> Set[int]:
        """Identifies atoms belonging to a residue starting from a given atom."""
        residue_atoms = set()
        stack = [start_atom.GetIdx()]

        while stack:
            atom_idx = stack.pop()
            if atom_idx not in residue_atoms and atom_idx not in visited_atoms:
                residue_atoms.add(atom_idx)
                atom = self.mol.GetAtomWithIdx(atom_idx)
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    bond = self.mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)
                    if not self._is_residue_boundary(bond):
                        if neighbor_idx not in visited_atoms and neighbor_idx not in residue_atoms:
                            stack.append(neighbor_idx)
        return residue_atoms

    def _merge_remaining_atoms(self, remaining_atoms: Set[int], all_residues: List[Set[int]]) -> None:
        """Merges remaining atoms with the closest residue."""
        for atom_idx in remaining_atoms:
            closest_residue = self._find_closest_residue(atom_idx, all_residues)
            if closest_residue is not None:
                closest_residue.add(atom_idx)

    def _find_closest_residue(self, atom_idx: int, residues: List[Set[int]]) -> Set[int]:
        """Finds the closest residue to a given atom."""
        atom = self.mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            for residue in residues:
                if neighbor.GetIdx() in residue:
                    return residue
        return None

    def _are_residues_connected(self, residue1: Set[int], residue2: Set[int]) -> bool:
        """Checks if two residues are connected."""
        return any(self.atom_graph.has_edge(atom1, atom2)
                   for atom1 in residue1
                   for atom2 in residue2)

    @staticmethod
    def _is_carbonyl_carbon(atom: Chem.Atom) -> bool:
        """Checks if an atom is a carbonyl carbon."""
        return (atom.GetSymbol() == 'C' and
                any(neighbor.GetSymbol() == 'O' and neighbor.GetTotalDegree() == 1
                    for neighbor in atom.GetNeighbors()))

    def _generate_2d_coordinates(self) -> Dict[int, Tuple[float, float]]:
        """Generates 2D coordinates for atoms in the molecule."""
        AllChem.Compute2DCoords(self.mol)
        return {atom.GetIdx(): (self.mol.GetConformer().GetAtomPosition(atom.GetIdx()).x,
                                self.mol.GetConformer().GetAtomPosition(atom.GetIdx()).y)
                for atom in self.mol.GetAtoms()}

    @staticmethod
    def _calculate_center_of_mass(positions: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculates the center of mass for a set of positions."""
        return tuple(map(lambda x: sum(x) / len(x), zip(*positions)))

    def _get_backbone_positions(self, residue_atoms: Set[int]) -> List[Tuple[float, float]]:
        """Gets the positions of backbone atoms in a residue."""
        backbone_atoms = self._get_backbone_atoms(residue_atoms)
        coordinates = self._generate_2d_coordinates()
        return [coordinates[idx] for idx in backbone_atoms]

    def draw_graphs(self) -> None:
        """Draws the atom graph, residue graph, and combined graph."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        self._draw_atom_graph(ax1)
        self._draw_residue_graph(ax2)
        self._draw_combined_graph(ax3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _get_cpk_color(atom_symbol: str) -> str:
        """Returns the CPK color for a given atom symbol."""
        cpk_colors = {
            'C': '#333333', 'N': '#3050F8', 'O': '#FF0D0D', 'H': '#FFFFFF',
            'S': '#FFFF30', 'Cl': '#1FF01F', 'B': '#F08080', 'P': '#FFA500',
            'Fe': '#FFA500', 'Ba': '#00C900', 'Na': '#AB5CF2'
        }
        return cpk_colors.get(atom_symbol, '#FFFFFF')

    @staticmethod
    def _get_label_color(background_color: str) -> str:
        """Determines the label color (black or white) based on background color brightness."""
        # Convert hex to RGB
        r = int(background_color[1:3], 16) / 255.0
        g = int(background_color[3:5], 16) / 255.0
        b = int(background_color[5:7], 16) / 255.0

        # Calculate luminance
        luminance = 0.299 * r + 0.587 * g + 0.114 * b

        # Use black for light backgrounds, white for dark backgrounds
        return '#000000' if luminance > 0.5 else '#FFFFFF'

    def _draw_atom_graph(self, ax: plt.Axes) -> None:
        """Draws the atom graph with adaptive label colors using scatter and text."""
        pos_atom = self._generate_2d_coordinates()
        node_colors = []
        label_colors = {}

        for node in self.atom_graph.nodes():
            atom_color = self._get_cpk_color(self.mol.GetAtomWithIdx(node).GetSymbol())
            node_colors.append(atom_color)
            label_colors[node] = self._get_label_color(atom_color)

        # Draw nodes using scatter
        node_positions = np.array([pos_atom[n] for n in self.atom_graph.nodes()])
        ax.scatter(node_positions[:, 0], node_positions[:, 1],
                   c=node_colors, s=200, zorder=2)

        # Draw edges using LineCollection
        edge_pos = [(pos_atom[e[0]], pos_atom[e[1]]) for e in self.atom_graph.edges()]
        edge_collection = mc.LineCollection(edge_pos, colors='k', linewidths=2, zorder=1)
        ax.add_collection(edge_collection)

        # Draw labels using matplotlib's text
        for node in self.atom_graph.nodes():
            x, y = pos_atom[node]
            label = str(node)
            ax.text(x, y, label, fontsize=8, color=label_colors[node],
                    ha='center', va='center', zorder=3, fontweight='bold')

        ax.set_title("Atom Graph (with advanced features)")
        ax.axis('off')

    def _draw_residue_graph(self, ax: plt.Axes) -> None:
        """Draws the residue graph using scatter and text."""
        residue_pos = {}
        for node in self.residue_graph.nodes():
            atoms = self.residue_graph.nodes[node]['atoms']
            backbone_positions = self._get_backbone_positions(atoms)
            residue_pos[node] = self._calculate_center_of_mass(backbone_positions)

        # Draw nodes using scatter
        node_positions = np.array([residue_pos[n] for n in self.residue_graph.nodes()])
        ax.scatter(node_positions[:, 0], node_positions[:, 1],
                   c='lightblue', s=1000, edgecolors='k', linewidths=0, zorder=2)

        # Draw edges using LineCollection
        edge_pos = [(residue_pos[e[0]], residue_pos[e[1]]) for e in self.residue_graph.edges()]
        edge_collection = mc.LineCollection(edge_pos, colors='lightblue', linewidths=3, zorder=1)
        ax.add_collection(edge_collection)

        # Draw labels using matplotlib's text
        for node in self.residue_graph.nodes():
            x, y = residue_pos[node]
            label = str(node)
            ax.text(x, y, label, fontsize=10, fontweight='bold', color='k',
                    ha='center', va='center', zorder=3)

        ax.set_title("Residue Hypergraph")
        ax.axis('off')

    def _draw_combined_graph(self, ax: plt.Axes) -> None:
        """Draws a combined graph with atoms, residues, and global node, layered appropriately."""
        combined_graph = nx.Graph()

        # Add atom nodes and edges
        atom_pos = self._generate_2d_coordinates()
        for atom in self.mol.GetAtoms():
            atom_color = self._get_cpk_color(atom.GetSymbol())
            label_color = self._get_label_color(atom_color)
            combined_graph.add_node(atom.GetIdx(), color=atom_color, label_color=label_color, node_type='atom', zorder=3)
        for bond in self.mol.GetBonds():
            combined_graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), color='k', edge_type='bond', zorder=2)

        # Add residue nodes and edges
        residue_pos = {}
        for i, (node, data) in enumerate(self.residue_graph.nodes(data=True)):
            atoms = data['atoms']
            backbone_positions = self._get_backbone_positions(atoms)
            residue_pos[node] = self._calculate_center_of_mass(backbone_positions)
            combined_graph.add_node(node, color='lightblue', label_color='black', alpha=1, node_type='residue', zorder=2)

            # Add semitransparent edges between residue and its atoms
            for atom in atoms:
                combined_graph.add_edge(node, atom, color='gray', alpha=1, style='dashed', edge_type='residue-atom', zorder=1)

        # Add edges between connected residues
        for edge in self.residue_graph.edges():
            combined_graph.add_edge(edge[0], edge[1], color='lightblue', style='solid', edge_type='residue-residue', zorder=1)

        # Add global node
        global_pos = self._calculate_center_of_mass(list(residue_pos.values()))
        combined_graph.add_node('Global', color='lightgreen', label_color='black', alpha=1, node_type='global', zorder=1)
        for node in self.residue_graph.nodes():
            combined_graph.add_edge('Global', node, color='gray', alpha=1, style='dashed', edge_type='global-residue', zorder=0)

        # Combine positions
        pos = {**atom_pos, **residue_pos, 'Global': global_pos}

        # Draw edges using LineCollection
        edge_types = ['global-residue', 'residue-residue', 'residue-atom', 'bond']
        for edge_type in edge_types:
            edges = [e for e in combined_graph.edges(data=True) if e[2]['edge_type'] == edge_type]
            if edges:
                edge_pos = [(pos[e[0]], pos[e[1]]) for e in edges]
                edge_collection = mc.LineCollection(edge_pos,
                                                    colors=[e[2]['color'] for e in edges],
                                                    alpha=[e[2].get('alpha', 1.0) for e in edges],
                                                    linestyles=[e[2].get('style', 'solid') for e in edges],
                                                    linewidths=1.5 if edge_type == "global-residue" else 1.5 if edge_type == "residue-atom" else 3 if edge_type == "residue-residue" else 2,
                                                    zorder=edges[0][2]['zorder'])
                ax.add_collection(edge_collection)

        # Draw nodes using scatter
        node_types = ['global', 'residue', 'atom']
        for node_type in node_types:
            nodes = [n for n, data in combined_graph.nodes(data=True) if data['node_type'] == node_type]
            if nodes:
                node_colors = [combined_graph.nodes[n]['color'] for n in nodes]
                node_sizes = [2000 if node_type == 'global' else 1000 if node_type == 'residue' else 200 for _ in nodes]
                node_positions = np.array([pos[n] for n in nodes])
                ax.scatter(node_positions[:, 0], node_positions[:, 1],
                        c=node_colors, s=node_sizes,
                        alpha=combined_graph.nodes[nodes[0]].get('alpha', 1.0),
                        zorder=combined_graph.nodes[nodes[0]]['zorder'])

        # Draw labels using matplotlib's text
        for node_type in node_types:
            nodes = [n for n, data in combined_graph.nodes(data=True) if data['node_type'] == node_type]
            font_size = 12 if node_type == 'global' else 10 if node_type == 'residue' else 8
            font_weight = 'bold'
            for node in nodes:
                x, y = pos[node]
                label = str(node)
                font_color = combined_graph.nodes[node]['label_color']
                ax.text(x, y, label, fontsize=font_size, fontweight=font_weight,
                        color=font_color, ha='center', va='center',
                        zorder=combined_graph.nodes[node]['zorder'] + 1)

        ax.set_title("Complete Graph")
        ax.axis('off')

    def to_pytorch_geometric(self):
        """Converts the molecular graph to PyTorch Geometric format with enhanced features."""
        # Convert atom features to node features
        node_features = []
        for node, data in self.atom_graph.nodes(data=True):
            features = [
                # Basic properties
                data['atomic_num'],
                data['degree'],
                data['total_degree'],
                data['explicit_valence'],
                data['implicit_valence'],
                data['formal_charge'],
                data['hybridization'],
                data['total_num_Hs'],
                data['num_explicit_Hs'],
                data['num_implicit_Hs'],
                data['atomic_mass'],
                data['isotope'],

                # Electronic properties
                data['gasteiger_charge'],
                data['default_valency'],
                data['num_radical_electrons'],

                # Topological properties
                data['is_aromatic'],
                data['is_in_ring'],
                data['is_in_ring_size_3'],
                data['is_in_ring_size_4'],
                data['is_in_ring_size_5'],
                data['is_in_ring_size_6'],
                data['is_in_ring_size_7'],
                data['num_rings'],

                # Stereochemistry
                data['chiral_tag'],
                data['has_chiral_tag'],

                # Contributions to molecular properties
                data['crippen_contrib_logp'],
                data['crippen_contrib_mr'],
                data['tpsa_contrib'],
                data['labute_asa'],

                # Environment features
                data['num_neighbors'],
                data['num_aromatic_bonds'],
                data['num_single_bonds'],
                data['num_double_bonds'],
                data['num_triple_bonds'],

                # Protein-specific features
                data['is_backbone'],
                data['is_sidechain'],
                data['is_terminal']
            ]
            node_features.append(features)
        node_features = torch.tensor(node_features, dtype=torch.float)

        # Convert bonds to edge indices and features
        edge_index = []
        edge_features = []
        for u, v, data in self.atom_graph.edges(data=True):
            # Add both directions for undirected graph
            edge_index.extend([[u, v], [v, u]])

            features = [
                # Basic properties
                data['bond_type'],
                data['is_conjugated'],
                data['is_in_ring'],
                data['is_aromatic'],

                # Stereochemistry
                data['stereo'],
                data['is_stereochem'],

                # Topology
                data['is_rotatable'],
                data['ring_size'],

                # Bond environment
                data['valence_contribution'],
                data['is_conjugated_to_aromatic'],

                # Peptide-specific features
                data['is_amide'],
                data['is_peptoid'],
                data['is_disulfide'],
                data['is_backbone'],

                # Atom context
                data['connects_aromatic'],
                data['connects_ring'],
                data['connects_different_rings']
            ]
            # Add features for both directions
            edge_features.extend([features, features])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_features = torch.tensor(edge_features, dtype=torch.float)

        # Enhanced residue features
        residue_features = []
        for node, data in self.residue_graph.nodes(data=True):
            atoms = data['atoms']
            backbone_atoms = data['backbone_atoms']

            # Calculate additional residue-level features
            aromatic_atoms = sum(1 for atom_idx in atoms
                            if self.mol.GetAtomWithIdx(atom_idx).GetIsAromatic())
            ring_atoms = sum(1 for atom_idx in atoms
                            if self.mol.GetAtomWithIdx(atom_idx).IsInRing())
            charged_atoms = sum(1 for atom_idx in atoms
                            if self.mol.GetAtomWithIdx(atom_idx).GetFormalCharge() != 0)

            features = [
                len(atoms),                    # Total atoms in residue
                len(backbone_atoms),           # Number of backbone atoms
                aromatic_atoms,                # Number of aromatic atoms
                ring_atoms,                    # Number of atoms in rings
                charged_atoms,                 # Number of charged atoms
                len(atoms - backbone_atoms),   # Number of side chain atoms
                float(len(backbone_atoms)) / len(atoms)  # Backbone ratio
            ]
            residue_features.append(features)
        residue_features = torch.tensor(residue_features, dtype=torch.float)

        # Enhanced global features
        global_features = torch.tensor([
            self.peptide.num_residues,
            self.peptide.num_residues_in_main_loop,
            self.peptide.num_residues_in_branch,
            self.mol.GetNumAtoms(),           # Total number of atoms
            self.mol.GetNumBonds(),           # Total number of bonds
            rdMolDescriptors.CalcNumRings(self.mol),  # Total number of rings
            rdMolDescriptors.CalcNumRotatableBonds(self.mol),  # Number of rotatable bonds
            rdMolDescriptors.CalcNumAromaticRings(self.mol),   # Number of aromatic rings
            float(len([a for a in self.mol.GetAtoms() if a.GetIsAromatic()])) / self.mol.GetNumAtoms(),  # Aromatic ratio
            Crippen.MolLogP(self.mol),        # Molecular LogP
            rdMolDescriptors.CalcTPSA(self.mol)  # Total polar surface area
        ], dtype=torch.float).unsqueeze(0)

        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            residue_x=residue_features,
            global_x=global_features
        )
