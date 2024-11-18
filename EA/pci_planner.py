# pci_planner.py

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import torch

# Determine the device to use: GPU if available, else CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

class DataLoader:
    """Class responsible for loading data from Excel files."""

    def __init__(self, cell_info_file, conflict_inference_file, confusion_file):
        self.cell_info_file = cell_info_file
        self.conflict_inference_file = conflict_inference_file
        self.confusion_file = confusion_file

    def load_data(self):
        """Load Excel files into DataFrames"""
        df1 = pd.read_excel(self.cell_info_file)
        df2 = pd.read_excel(self.conflict_inference_file)
        df3 = pd.read_excel(self.confusion_file)
        return df1, df2, df3

class MatrixConstructor:
    """Class responsible for constructing A, B, C"""

    def __init__(self, df1, df2, df3, num_pcis=1008, num_cells=2890):
        self.df1 = df1
        self.df2 = df2
        self.df3 = df3
        self.num_pcis = num_pcis
        self.num_cells = num_cells

    def construct_matrices(self):
        """Construct and return sparse matrices A, B, C."""
        A_dense = np.zeros((self.num_cells, self.num_cells), dtype=int)  # Collision Matrix
        B_dense = np.zeros((self.num_cells, self.num_cells), dtype=int)  # Confusion Matrix
        C_dense = np.zeros((self.num_cells, self.num_cells), dtype=int)  # Interference Matrix

        # Construct matrices A and C from df2
        for _, row in self.df2.iterrows():
            i, j = int(row['Index_小区编号']), int(row['Index_邻小区编号'])
            A_dense[i][j] = row['冲突MR数']
            C_dense[i][j] = row['干扰MR数']
        
        # Construct matrix B from df3
        for _, row in self.df3.iterrows():
            i, j = int(row['Index_小区0编号']), int(row['Index_小区1编号'])
            B_dense[i][j] = row['混淆MR数']
            B_dense[j][i] = row['混淆MR数']

        # Convert dense matrices to sparse format
        A = csr_matrix(A_dense)
        B = csr_matrix(B_dense)
        C = csr_matrix(C_dense)

        return A, B, C
    
class FitnessCalculator:
    """Class responsible for calculating MR sums."""

    def __init__(self, A, B, C, num_pcis=1008):
        self.num_pcis = num_pcis
        self.device = device
        # Convert A, B, C to torch sparse tensors and move to device
        self.A = self.scipy_sparse_to_torch_sparse(A).to(self.device)
        self.B = self.scipy_sparse_to_torch_sparse(B).to(self.device)
        self.C = self.scipy_sparse_to_torch_sparse(C).to(self.device)
    
    def scipy_sparse_to_torch_sparse(self, scipy_mat):
        coo = scipy_mat.tocoo()
        # Convert coo.row and coo.col to a single NumPy array to avoid the warning
        indices = np.vstack((coo.row, coo.col))
        indices = torch.from_numpy(indices).long()
        values = torch.from_numpy(coo.data).float()
        shape = coo.shape
        torch_sparse_mat = torch.sparse_coo_tensor(indices, values, torch.Size(shape))
        return torch_sparse_mat

    def calculate_MR(self, pci_assignment):
        """Calculate and return MR sums given a PCI assignment"""
        num_cells = len(pci_assignment)

        # Convert pci_assignment to torch tensor and move to device
        pci_assignment = torch.tensor(pci_assignment, dtype=torch.long, device=self.device)

        # Initialize X and Y based on the PCI assignment
        X = torch.zeros((num_cells, self.num_pcis), dtype=torch.float32, device=self.device)
        Y = torch.zeros((num_cells, 3), dtype=torch.float32, device=self.device)

        indices = torch.arange(num_cells, device=self.device)

        X[indices, pci_assignment] = 1
        Y[indices, pci_assignment % 3] = 1

        # Compute M and M_mod3
        M = torch.mm(X, X.t())
        M_mod3 = torch.mm(Y, Y.t())

        # Compute the collision
        indices_A = self.A.coalesce().indices()
        values_A = self.A.coalesce().values()
        M_selected = M[indices_A[0], indices_A[1]]
        collision_sum = torch.sum(values_A * M_selected)

        # Compute confusion (divided by 2 if B is symmetric to avoid double-counting)
        indices_B = self.B.coalesce().indices()
        values_B = self.B.coalesce().values()
        M_selected = M[indices_B[0], indices_B[1]]
        confusion_sum = torch.sum(values_B * M_selected) / 2

        # Compute the interference
        indices_C = self.C.coalesce().indices()
        values_C = self.C.coalesce().values()
        M_mod3_selected = M_mod3[indices_C[0], indices_C[1]]
        interference_sum = torch.sum(values_C * M_mod3_selected)

        MR_sum = collision_sum + confusion_sum + interference_sum

        # Extract scalar values
        collision_sum = collision_sum.item()
        confusion_sum = confusion_sum.item()
        interference_sum = interference_sum.item()
        MR_sum = MR_sum.item()

        return MR_sum, collision_sum, confusion_sum, interference_sum