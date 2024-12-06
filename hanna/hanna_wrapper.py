import sys
import os

# Add the 'hanna' directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import pickle
import torch

import numpy as np

from hanna.utils.HANNA import HANNA
from hanna.utils.Utils import predict, create_embedding_matrix
from hanna.utils.Utils import initiliaze_ChemBERTA


class HANNAWrapper:
    def __init__(self):
        # Paths for model and scaler
        model_path = os.path.join(os.getcwd(), 'hanna', 'Model', 'HANNA_Val.pt')
        scaler_path = os.path.join(os.getcwd(), 'hanna', 'Model', 'scalerHANNA_Val.pkl')
        # Load the model
        self.device = torch.device("cpu")
        self.model = HANNA().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        # Set the model to evaluation mode
        self.model.eval()
        # Load the scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # Initialize ChemBERTa
        self.ChemBERTA, self.tokenizer = initiliaze_ChemBERTA(model_name="DeepChem/ChemBERTa-77M-MTR", device=None)

    def compute_activity_coefficient(self, molar_fractions, index, temperature: float, SMILES_1: str = "CCCCO", SMILES_2: str = "O"):
        # NOTE: The warning "Some weights of RobertaModel were not initialized from the model checkpoint..." is expected and can be ignored, because we are not using the pooler head of the model.
        if len(molar_fractions) > 2:
            raise Exception('HANNA works by now only for 2 components!')

        x1_values = np.array([molar_fractions[0]])  # Vector of mole fractions of component 1
        embedding_matrix = create_embedding_matrix(SMILES_1, SMILES_2, temperature, self.device, self.ChemBERTA, self.tokenizer, x1_values) # Create the embedding matrix
        x_pred, ln_gammas_pred = predict(embedding_matrix, self.scaler, self.model, self.device)  # Predict the logarithmic activity coefficients
        gamma_index = np.exp(ln_gammas_pred)[0][index]

        return gamma_index
