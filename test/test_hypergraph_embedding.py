from ..core.peptide import Peptide

oxytocin = "CC[C@H](C)[C@H]1C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@@H](CSSC[C@@H](C(=O)N[C@H](C(=O)N1)CC2=CC=C(C=C2)O)N)C(=O)N3CCC[C@H]3C(=O)N[C@@H](CC(C)C)C(=O)NCC(=O)N)CC(=O)N)CCC(=O)N"
oxytocin_num_residues = 9
oxytocin_num_residues_in_main_loop = 6

peptide = Peptide(oxytocin, oxytocin_num_residues, oxytocin_num_residues_in_main_loop)
peptide.draw_graphs()
