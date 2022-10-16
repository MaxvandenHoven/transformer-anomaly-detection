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
    mask_prop: float = 0.3,
    epochs: int = 100,
    device: str = "cuda",
):    

    criterion = nn.MSELoss(reduction="sum") # sum

    for epoch in range(epochs):

        print(f"Epoch {epoch + 1}:")

        model.train()

        running_loss = 0
        num_samples = 0

        for batch in tqdm(train_data, desc="Training ..."):
            optimizer.zero_grad()

            mask = (torch.FloatTensor(batch.shape).uniform_() > mask_prop).to(device)
            batch_mask = batch * mask

            batch_pred, attn_weights = model(batch_mask)

            loss = criterion(~mask * batch_pred, ~mask * batch)
            # loss = criterion(batch_pred, batch)

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            # for param in model.parameters():
            #     if param.grad is not None:
            #         print(torch.norm(param.grad))


            optimizer.step()

            running_loss += loss.item() / (~mask).sum()
            num_samples += len(batch)


        train_loss = running_loss / num_samples


        model.eval()

        with torch.no_grad():
            running_loss = 0
            num_samples = 0

            for batch in tqdm(valid_data, desc="Validating ..."):
                mask = (torch.FloatTensor(batch.shape).uniform_() > mask_prop).to(device)
                batch_mask = batch * mask

                batch_pred, attn_weights = model(batch_mask)

                loss = criterion(~mask * batch_pred, ~mask * batch)
                # loss = criterion(batch_pred, batch)

                running_loss += loss.item() / (~mask).sum()
                num_samples += len(batch)

            valid_loss = running_loss / num_samples

        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}, Train loss: {train_loss}, Valid loss: {valid_loss}")

