import torch
import torch.nn as nn
from typing import List, Sequence, Tuple, Optional

###############################################################################
#  Base classes
###############################################################################

class BaseDayProjection(nn.Module):
    """Base class for day-specific projection functionality."""
    
    def __init__(
        self,
        neural_dim: int,
        n_days: int,
        input_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.neural_dim = neural_dim
        self.n_days = n_days
        
        # Day‑specific linear layers; identity initialisation helps
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_activation = nn.Softsign()
        self.day_dropout = nn.Dropout(input_dropout)
    
    def apply_day_projection(
        self, 
        x: torch.Tensor, 
        day_idx: torch.Tensor
    ) -> torch.Tensor:
        """Apply day-specific projection to input tensor."""
        weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)
        x = torch.einsum("btd,bdk->btk", x, weights) + biases
        x = self.day_activation(x)
        x = self.day_dropout(x)
        return x


class BasePatchModule(nn.Module):
    """Base class for patch/sliding window functionality."""
    
    def __init__(
        self,
        neural_dim: int,
        patch_size: int = 0,
        patch_stride: int = 1,
    ) -> None:
        super().__init__()
        self.neural_dim = neural_dim
        self.patch_size = patch_size
        self.patch_stride = patch_stride
    
    def apply_patch(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sliding window concatenation if patch_size > 0."""
        if self.patch_size <= 1:
            return x
            
        # Input shape: (B, T, D) -> (B, D, 1, T)
        x_unfold = x.unsqueeze(1).permute(0, 3, 1, 2)
        x_unfold = x_unfold.unfold(3, self.patch_size, self.patch_stride)
        # Remove dummy dimension and reshape to (B, num_patches, patch_size*D)
        x_unfold = x_unfold.squeeze(2).permute(0, 2, 3, 1)
        return x_unfold.reshape(x.size(0), x_unfold.size(1), -1)
    
    def get_input_size(self) -> int:
        """Get input size considering patch dimensions."""
        input_size = self.neural_dim
        if self.patch_size > 1:
            input_size *= self.patch_size
        return input_size


class BaseRNNDecoder(nn.Module):
    """Base class for RNN-based decoders with common functionality."""
    
    def __init__(
        self,
        neural_dim: int,
        n_units: int,
        n_days: int,
        n_classes: int,
        rnn_dropout: float = 0.0,
        input_dropout: float = 0.0,
        n_layers: int = 1,
        patch_size: int = 0,
        patch_stride: int = 1,
        bidirectional: bool = False,
        name: str = "BaseRNNDecoder",
    ) -> None:
        super().__init__()
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_days = n_days
        self.rnn_dropout = rnn_dropout
        self.input_dropout = input_dropout
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.bidirectional = bidirectional
        self.name = name
        
        # Initialize projection and patch modules
        self.projection = BaseDayProjection(
            neural_dim=neural_dim,
            n_days=n_days,
            input_dropout=input_dropout,
        )
        self.patch = BasePatchModule(
            neural_dim=neural_dim,
            patch_size=patch_size,
            patch_stride=patch_stride,
        )
    
    def _init_gru_weights(self, gru: nn.GRU) -> None:
        """Initialize GRU weights."""
        for name, param in gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
    
    def _create_initial_state(self, batch_size: int) -> torch.Tensor:
        """Create initial hidden state."""
        num_directions = 2 if self.bidirectional else 1
        return self.h0.expand(
            num_directions * self.n_layers, 
            batch_size, 
            self.n_units
        ).contiguous()


###############################################################################
#  Model definitions using base classes
###############################################################################

class GRUDecoder(BaseRNNDecoder):
    """A decoder composed of day‑specific linear layers followed by a GRU."""

    def __init__(
        self,
        neural_dim: int,
        n_units: int,
        n_days: int,
        n_classes: int,
        rnn_dropout: float = 0.0,
        input_dropout: float = 0.0,
        n_layers: int = 1,
        patch_size: int = 0,
        patch_stride: int = 1,
        name: str = "GRUDecoder",
    ) -> None:
        super().__init__(
            neural_dim=neural_dim,
            n_units=n_units,
            n_days=n_days,
            n_classes=n_classes,
            rnn_dropout=rnn_dropout,
            input_dropout=input_dropout,
            n_layers=n_layers,
            patch_size=patch_size,
            patch_stride=patch_stride,
            bidirectional=False,
            name=name,
        )
        
        # GRU layer
        input_size = self.patch.get_input_size()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=self.n_units,
            num_layers=self.n_layers,
            dropout=self.rnn_dropout if self.n_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self._init_gru_weights(self.gru)
        
        # Output layer
        self.out = nn.Linear(self.n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)
        
        # Learnable initial hidden state
        self.h0 = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(self.n_layers, 1, self.n_units))
        )

    def forward(
        self,
        x: torch.Tensor,
        day_idx: torch.Tensor,
        states: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # Apply day-specific projection
        x = self.projection.apply_day_projection(x, day_idx)
        
        # Apply patch if needed
        x = self.patch.apply_patch(x)
        
        # Prepare initial hidden state
        if states is None:
            states = self._create_initial_state(x.size(0))
        
        # Run through GRU
        output, hidden = self.gru(x, states)
        logits = self.out(output)
        
        return (logits, hidden) if return_state else logits


class BiGRUEncoder(BaseRNNDecoder):
    """Bidirectional GRU encoder for CTC-based sequence modeling."""

    def __init__(
        self,
        neural_dim: int,
        n_units: int,
        n_days: int,
        n_classes: int,
        rnn_dropout: float = 0.0,
        input_dropout: float = 0.0,
        n_layers: int = 1,
        patch_size: int = 0,
        patch_stride: int = 1,
        name: str = "BiGRUEncoder",
    ) -> None:
        super().__init__(
            neural_dim=neural_dim,
            n_units=n_units,
            n_days=n_days,
            n_classes=n_classes,
            rnn_dropout=rnn_dropout,
            input_dropout=input_dropout,
            n_layers=n_layers,
            patch_size=patch_size,
            patch_stride=patch_stride,
            bidirectional=True,
            name=name,
        )
        
        # Bidirectional GRU
        input_size = self.patch.get_input_size()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=self.n_units,
            num_layers=self.n_layers,
            dropout=self.rnn_dropout if self.n_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self._init_gru_weights(self.gru)
        
        # Output head (2 * hidden size for BiGRU)
        self.out = nn.Linear(2 * self.n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)
        
        # Learnable initial hidden state (layers × directions)
        self.h0 = nn.Parameter(torch.zeros(2 * self.n_layers, 1, self.n_units))
        nn.init.xavier_uniform_(self.h0)

    def forward(
        self,
        x: torch.Tensor,
        day_idx: torch.Tensor,
        states: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # Apply day-specific projection
        x = self.projection.apply_day_projection(x, day_idx)
        
        # Apply patch if needed
        x = self.patch.apply_patch(x)
        
        # Prepare initial hidden state
        if states is None:
            states = self._create_initial_state(x.size(0))
        
        # Run through BiGRU
        output, hidden = self.gru(x, states)
        logits = self.out(output)
        
        return (logits, hidden) if return_state else logits


class CNNGRUDecoder(nn.Module):
    """A decoder that prepends a 1D convolutional block before the GRU."""

    def __init__(
        self,
        neural_dim: int,
        n_units: int,
        n_days: int,
        n_classes: int,
        n_layers: int = 1,
        dropout: float = 0.0,
        cnn_kernel: int = 3,
        cnn_stride: int = 1,
        name: str = "CNNGRUDecoder",
    ) -> None:
        super().__init__()
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_days = n_days
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout_prob = dropout
        self.cnn_stride = cnn_stride
        self.name = name
        
        # Day-specific projection (reusing BaseDayProjection)
        self.projection = BaseDayProjection(
            neural_dim=neural_dim,
            n_days=n_days,
            input_dropout=dropout,
        )
        
        # 1D convolutional block
        self.conv1 = nn.Conv1d(
            neural_dim,
            neural_dim,
            kernel_size=cnn_kernel,
            stride=cnn_stride,
            padding=cnn_kernel // 2,
        )
        self.bn1 = nn.BatchNorm1d(neural_dim)
        self.relu = nn.ReLU()
        
        # GRU backbone
        self.gru = nn.GRU(
            neural_dim,
            n_units,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        
        # Output classifier
        self.fc = nn.Linear(n_units, n_classes)

    def forward(self, x: torch.Tensor, day_idx: torch.Tensor) -> torch.Tensor:
        # Day-specific projection
        x = self.projection.apply_day_projection(x, day_idx)
        
        # Convolution expects [B, C, T]
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        
        # GRU
        x, _ = self.gru(x)
        # Classifier
        logits = self.fc(x)
        return logits