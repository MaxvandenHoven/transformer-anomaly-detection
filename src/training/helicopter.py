import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from ..models.transformer import CustomTransformer


def train_helicopter(
    train_data: DataLoader,
    valid_data: DataLoader,
    model: CustomTransformer,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler,
    mask_strategy: str,
    loss_strategy: str,
    random_mask_prop: float = 0.3,
    noise_std: float = 0.1,
    epochs: int = 100,
    checkpoints: int = 5,
    save_prefix: str = "model",
    device: str = "cuda",
):    
    """_summary_

    Args:
        train_data (DataLoader): Training data dataloader
        valid_data (DataLoader): Validation data dataloader
        model (CustomTransformer): Model (must be on `device`)
        optimizer (optim.Optimizer): Optimization algorithm
        scheduler (optim.lr_scheduler): Learning rate scheduler
        mask_strategy (str): Controls how masks are generated:
            - "random zero": random indices are set to zero
            - "contiguous zero": random contiguous indices are set to zero
            - "random noise": random indices have added noise
            - "contiguous noise": random contigous indices have added noise
        loss_strategy (str): Controls how loss is calculated:
            - "complete": loss is computed over all inputs
            - "mask": loss is computed over masked inputs only
        random_mask_prop (float, optional): Probability that a point is masked. Only used 
            when mask_strategy contains "random". Defaults to 0.3.
        noise_std (float, optional): Variance of Gaussian noise added to masked 
            inputs. Only used when mask_strategy contains "noise".
        epochs (int, optional): Number of epochs to train for. Defaults to 100.
        checkpoint (int, optional): Checkpoint save interval.
        save_prefix (str, optional): Prefix to identify model checkpoints.
        device (str, optional): Device to train on. Defaults to "cuda".
    """
    # Check configuration
    assert mask_strategy in ["random zero", "contiguous zero", "random noise", "contiguous noise"]
    assert loss_strategy in ["complete", "mask"]

    # If loss_strategy is mask, use non-averaged MSE loss and instead divide by number
    # of masked elements to get loss over masked elements only
    if loss_strategy == "mask":
        criterion = nn.MSELoss(reduction="sum")
    elif loss_strategy == "complete":
        criterion = nn.MSELoss(reduction="mean")

    # Main training loop
    for epoch in range(epochs):
        # Loggin (TODO: tensorboard)
        print(f"Epoch {epoch + 1}:")
        
        # Set training flag for normalization behaviour
        model.train()

        # Track cumulative loss over epoch and number of samples to calculate average loss
        running_loss = 0
        num_samples = 0

        # Train batches
        for batch in tqdm(train_data, desc="Training ..."):
            # Move data from cpu to gpu. This is done in training loop to reduce gpu usage
            batch = batch.to(device)

            # Zero out gradients
            optimizer.zero_grad()

            # Generate binary mask
            if "random" in mask_strategy:
                mask = (torch.rand_like(batch) > random_mask_prop).to(device)
            elif "contiguous" in mask_strategy:
                mask = None # TODO

            # Apply mask
            if "zero" in mask_strategy:
                batch_mask = batch * mask # Elements in mask set to False become 0
            elif "noise" in mask_strategy:
                noise = (torch.randn_like(batch) * noise_std).to(device)
                noise_mask = noise * mask
                batch_mask = batch + noise_mask

            # Model attempts to reconstruct original input
            batch_pred, attn_weights = model(batch_mask)
            
            # Compute appropriate loss (mask strategy only considers values where mask is False)
            if loss_strategy == "mask":
                loss = criterion(~mask * batch_pred, ~mask * batch)
            elif loss_strategy == "complete":
                loss = criterion(batch_pred, batch)

            # Compute gradients
            loss.backward()

            # Perform gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            # Optional debugging
            # for param in model.parameters():
            #     if param.grad is not None:
            #         print(torch.norm(param.grad))

            # Update weights
            optimizer.step()

            # Update running totals
            if loss_strategy == "mask":
                running_loss += loss.item() / (~mask).sum()
            elif loss_strategy == "complete":
                running_loss += loss.item()
            num_samples += len(batch)

        # Compute average loss over training samples
        train_loss = running_loss / num_samples

        # Evaluation flag to stop parameter updates
        model.eval()

        # Disable gradients
        with torch.no_grad():
            # Cumulative totals for average loss computations
            running_loss = 0
            num_samples = 0

            for batch in tqdm(valid_data, desc="Validating ..."):
                # Move batch to gpu
                batch = batch.to(device)

                # Generate binary mask
                if "random" in mask_strategy:
                    mask = (torch.rand_like(batch) > random_mask_prop).to(device)
                elif "contiguous" in mask_strategy:
                    mask = None # TODO

                # Apply mask
                if "zero" in mask_strategy:
                    batch_mask = batch * mask # Elements in mask set to False become 0
                elif "noise" in mask_strategy:
                    noise = (torch.randn_like(batch) * noise_std).to(device)
                    noise_mask = noise * mask
                    batch_mask = batch + noise_mask

                # Model attempts to reconstruct original input
                batch_pred, attn_weights = model(batch_mask)

                # Compute appropriate loss
                if loss_strategy == "mask":
                    loss = criterion(~mask * batch_pred, ~mask * batch)
                elif loss_strategy == "complete":
                    loss = criterion(batch_pred, batch)

                # Update running totals
                if loss_strategy == "mask":
                    running_loss += loss.item() / (~mask).sum()
                elif loss_strategy == "complete":
                    running_loss += loss.item()
                num_samples += len(batch)

            # Compute average loss over validation samples
            valid_loss = running_loss / num_samples

        # Generate new optimizer parameters for next epoch
        scheduler.step()

        # Tracking (TODO: tensorboard)
        print(f"Epoch {epoch+1}/{epochs}, Train loss: {train_loss}, Valid loss: {valid_loss}")

        if (epoch + 1) % checkpoints == 0:
            torch.save(model, f"{save_prefix}-helicopter-{mask_strategy}-{loss_strategy}-{epoch + 1}.save")

