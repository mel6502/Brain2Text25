import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from CTC_decode_acc_measurements import greedy_ctc_decode
from trainer import evaluate

class Inference:
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def predict(
        self,
        test_loader: torch.utils.data.DataLoader
    ) -> List:
        """Generate predictions for test data."""
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for xs, _, _, _, day_idxs in test_loader:
                xs = xs.to(self.device)
                day_idxs = day_idxs.to(self.device)
                
                logits = self.model(xs, day_idxs)
                predictions = greedy_ctc_decode(logits)
                all_predictions.extend(predictions)
        
        print(f"Generated predictions for {len(all_predictions)} test samples")
        return all_predictions
    
    def evaluate(
        self,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float, float]:
        """Evaluate model on validation data."""
        return evaluate(self.model, val_loader, criterion, decode=True)