import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# =============================================================================
# S4PRED Model Architecture (Adapted from s4pred-main/network.py)
# =============================================================================

class ResidueEmbedding(nn.Embedding):
    def __init__(self, vocab_size=22, embed_size=128, padding_idx=21):
        super().__init__(vocab_size, embed_size, padding_idx=padding_idx)

class GRUnet(nn.Module):
    def __init__(self, lstm_hdim=1024, embed_size=128, num_layers=3, bidirectional=True, outsize=3):
        super().__init__()
        """
            This version of the model has all the bells & whistles (e.g. 
            dropconnect) ripped out so its slimmed down for inference
        """
        self.lstm_hdim = lstm_hdim
        self.embed = ResidueEmbedding(vocab_size=22, embed_size=embed_size, padding_idx=21)
        self.lstm = nn.GRU(128, 1024, num_layers=3, bidirectional=True, batch_first=True, dropout=0.0)
        self.outlayer = nn.Linear(lstm_hdim * 2, outsize)
        self.finalact = F.log_softmax

    def forward(self, x):
        """
            Assumes a batch size of one currently but can be changed
        """
        x = self.embed(x)
        x, _ = self.lstm(x)
        x = self.outlayer(x)
        x = self.finalact(x, dim=-1)
        return x.squeeze()

class S4PRED(nn.Module):
    def __init__(self, single_model=False):
        super().__init__()
        """
            This loads the ensemble of models in a lazy way but its clear and 
            leaves the weight loading out of the run_model script. 
            
            Args:
                single_model (bool): If True, only initialize model_1 to save memory/space.
        """
        self.single_model = single_model
        
        # Manually listing for clarity and hot swapping in future
        self.model_1 = GRUnet()
        if not self.single_model:
            self.model_2 = GRUnet()
            self.model_3 = GRUnet()
            self.model_4 = GRUnet()
            self.model_5 = GRUnet()

    def forward(self, x):
        y_1 = self.model_1(x)
        
        if self.single_model:
            return y_1
            
        y_2 = self.model_2(x)
        y_3 = self.model_3(x)
        y_4 = self.model_4(x)
        y_5 = self.model_5(x)
        
        # Ensemble averaging
        y_out = y_1 * 0.2 + y_2 * 0.2 + y_3 * 0.2 + y_4 * 0.2 + y_5 * 0.2
        return y_out

# =============================================================================
# Structure Predictor Wrapper
# =============================================================================

class StructurePredictor:
    """
    Wrapper class for structure prediction using S4PRED.
    Manages model loading, input processing, and fallback heuristics.
    Supports both single model mode and ensemble mode.
    """
    def __init__(self, weights_dir=None, device=None, single_model_mode=True):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_model = False
        self.model = None
        self.single_model_mode = single_model_mode
        
        # Amino acid mapping (A-Z) -> Index (0-19), 20=Unknown/Pad
        # Based on s4pred-main/utilities.py aas2int function
        self.aa_to_idx = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 
            'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16,
            'W': 17, 'Y': 18, 'V': 19
        }
        
        # Heuristic propensities (fallback)
        self.propensities = {
            'A': 'H', 'R': 'H', 'N': 'C', 'D': 'C', 'C': 'E',
            'Q': 'H', 'E': 'H', 'G': 'C', 'H': 'H', 'I': 'E',
            'L': 'H', 'K': 'H', 'M': 'H', 'F': 'E', 'P': 'C',
            'S': 'C', 'T': 'E', 'W': 'E', 'Y': 'E', 'V': 'E'
        }

        # Try to load model if weights directory is provided
        if weights_dir and os.path.exists(weights_dir):
            try:
                self._load_model(weights_dir)
            except Exception as e:
                print(f"StructurePredictor: Failed to load model from {weights_dir}. Error: {e}")
                print("StructurePredictor: Falling back to heuristic mode.")
        else:
            if weights_dir:
                print(f"StructurePredictor: Weights directory not found at {weights_dir}")
            print("StructurePredictor: Using heuristic mode (propensity-based).")

    def _load_model(self, weights_dir):
        """
        Load S4PRED model(s).
        If single_model_mode is True, loads only weights_1.pt.
        Otherwise, loads the full 5-model ensemble.
        """
        self.model = S4PRED(single_model=self.single_model_mode)
        
        if self.single_model_mode:
            # Single model mode: only load weights_1.pt
            w_file = 'weights_1.pt'
            w_path = os.path.join(weights_dir, w_file)
            
            if os.path.exists(w_path):
                state_dict = torch.load(w_path, map_location=self.device)
                self.model.model_1.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                self.model.requires_grad = False
                self.use_model = True
                print(f"StructurePredictor: Successfully loaded S4PRED (Single Model) from {w_path}")
            else:
                raise FileNotFoundError(f"Weight file {w_file} not found in {weights_dir}")
        else:
            # Ensemble mode: load all 5 weights
            weight_files = [f'weights_{i}.pt' for i in range(1, 6)]
            models = [self.model.model_1, self.model.model_2, self.model.model_3, self.model.model_4, self.model.model_5]
            
            loaded_count = 0
            for i, w_file in enumerate(weight_files):
                w_path = os.path.join(weights_dir, w_file)
                if os.path.exists(w_path):
                    state_dict = torch.load(w_path, map_location=self.device)
                    models[i].load_state_dict(state_dict)
                    loaded_count += 1
                else:
                    raise FileNotFoundError(f"Weight file {w_file} not found in {weights_dir}")
            
            if loaded_count == 5:
                self.model.to(self.device)
                self.model.eval()
                self.model.requires_grad = False
                self.use_model = True
                print(f"StructurePredictor: Successfully loaded S4PRED ensemble from {weights_dir}")

    def predict(self, sequence):
        """
        Predict secondary structure for a given sequence.
        Returns:
            structure (str): String of 'H', 'E', 'C'
            helicity (float): Fraction of 'H' in the structure
        """
        if not sequence:
            return "", 0.0

        if self.use_model:
            return self._predict_model(sequence)
        else:
            return self._predict_heuristic(sequence)

    def _predict_model(self, sequence):
        # Convert sequence to indices (20 is unknown/pad)
        indices = [self.aa_to_idx.get(aa, 20) for aa in sequence]
        input_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            # S4PRED output is log_softmax, shape (seq_len, 3)
            log_probs = self.model(input_tensor)
            
            # Get class with max probability
            preds = log_probs.argmax(dim=-1).cpu().numpy()
        
        # Map 0->C, 1->H, 2->E (Based on run_model.py ind2char)
        # ind2char={0:"C", 1:"H", 2:"E"}
        mapping = {0: 'C', 1: 'H', 2: 'E'} 
        structure = "".join([mapping.get(p, 'C') for p in preds])
        
        helicity = structure.count('H') / len(structure)
        return structure, helicity

    def _predict_heuristic(self, sequence):
        """
        Simple propensity-based prediction + smoothing.
        """
        # Step 1: Assign initial state based on propensity
        raw_states = [self.propensities.get(aa, 'C') for aa in sequence]
        
        # Step 2: Simple smoothing (window of 3)
        smoothed_states = raw_states.copy()
        n = len(sequence)
        if n >= 3:
            for i in range(1, n-1):
                prev_s = raw_states[i-1]
                next_s = raw_states[i+1]
                if prev_s == next_s:
                    smoothed_states[i] = prev_s
        
        structure = "".join(smoothed_states)
        helicity = structure.count('H') / len(structure)
        return structure, helicity

if __name__ == "__main__":
    # Test
    # Assuming weights might be in a 'weights' folder relative to this script for testing
    # For now, it will likely fall back to heuristic
    predictor = StructurePredictor(weights_dir="s4pred_weights") 
    seq = "MYVCVCNGITEEMLDTAQKQGLSDREILNRLGVGNSCGVCVIDALDNMRSNSLKSQKTSNRKDSKKS"
    struct, score = predictor.predict(seq)
    print(f"Sequence: {seq}")
    print(f"Structure: {struct}")
    print(f"Helicity: {score:.2f}")
