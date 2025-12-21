import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from data_loader import load_and_prepare_data, create_dataloaders
from trainer import Trainer
from inference import Inference
from utils import load_data, merge_sessions, prepare_split_dataset
from data_loader import collate_fn

def train_and_evaluate(
    model_class: type[nn.Module],
    model_params: Dict[str, Any],
    n_epochs: int = 1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    batch_size: int = 8,
    device: torch.device | None = None,
    max_train_trials: int | None = None,
    max_val_trials: int | None = None,
    max_test_trials: int | None = None,
    decode_val: bool = True,
    checkpoint_dir: str = "BaselineGRU"
) -> Tuple[List, nn.Module, Dict[str, List]]:
    """High‑level routine to train a model and report performance."""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load data
    print("Loading data...")
    train_examples, val_examples, test_examples = load_and_prepare_data(
        max_train_trials=max_train_trials,
        max_val_trials=max_val_trials,
        max_test_trials=max_test_trials
    )
    
    # 2. Instantiate model
    print("Initializing model...")
    model = model_class(**model_params)
    model.to(device)
    
    # 3. Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_examples, val_examples, test_examples,
        model, batch_size, device
    )
    
    # 4. Train model
    print("Starting training...")
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        checkpoint_dir=checkpoint_dir
    )
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        decode_val=decode_val
    )
    
    # 5. Load best model for inference
    print("\nLoading best model for inference...")
    trainer.load_checkpoint("best_model.ckpt")
    
    # 6. Generate predictions
    print("Generating predictions...")
    inference = Inference(model, device)
    test_predictions = inference.predict(test_loader)
    
    return test_predictions, model, history


def train_only(
    model_class: type[nn.Module],
    model_params: Dict[str, Any],
    n_epochs: int = 1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    batch_size: int = 8,
    device: torch.device | None = None,
    max_train_trials: int | None = None,
    max_val_trials: int | None = None,
    checkpoint_dir: str = "BaselineGRU"
) -> Tuple[nn.Module, Dict[str, List]]:
    """Train a model without inference."""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load only train and validation data
    raw_data = load_data()
    merged = merge_sessions(raw_data)
    
    train_examples = prepare_split_dataset(
        merged["train"], drop_blank=True, max_trials=max_train_trials, has_labels=True
    )
    val_examples = prepare_split_dataset(
        merged.get("val", []), drop_blank=True, max_trials=max_val_trials, has_labels=True
    )
    
    # Instantiate model
    model = model_class(**model_params)
    model.to(device)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_examples,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, model, device),
    )
    val_loader = torch.utils.data.DataLoader(
        val_examples,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, model, device),
    )
    
    # Train
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        checkpoint_dir=checkpoint_dir
    )
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        decode_val=True
    )
    
    return model, history


def inference_only(
    model_class: type[nn.Module],
    model_params: Dict[str, Any],
    checkpoint_path: str = "BaselineGRU/best_model.ckpt",
    batch_size: int = 8,
    device: torch.device | None = None,
    max_test_trials: int | None = None
) -> List:
    """Load a trained model and run inference."""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data only
    raw_data = load_data()
    merged = merge_sessions(raw_data)
    
    test_examples = prepare_split_dataset(
        merged.get("test", []), drop_blank=True, max_trials=max_test_trials, has_labels=False
    )
    
    # Instantiate and load model
    model = model_class(**model_params)
    model.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    
    # Create test dataloader
    test_loader = torch.utils.data.DataLoader(
        test_examples,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, model, device),
    )
    
    # Run inference
    inference = Inference(model, device)
    predictions = inference.predict(test_loader)
    
    return predictions