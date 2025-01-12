import torch
import torch.nn as nn

from tqdm import tqdm


def train(data_loader, model, optimizer, device):
    """
    This function does training for one epoch
    :param data_loader: this is the pytorch dataloader
    :param model: pytorch model
    :param optimizer: optimizer, for e.g. adam, sgd, etc
    :param device: cuda/cpu
    """
    # put the model in train mode
    model.train()
    running_loss = 0.0
    total_batches = len(data_loader)
    progress_bar = tqdm(data_loader, total=total_batches, desc="Training")

    # go over every batch of data in data loader
    for batch_idx, data in enumerate(progress_bar):
        # remember, we have image and targets
        # in our dataset class
        inputs = data["image"]
        targets = data["targets"]

        # move inputs/targets to cuda/cpu device
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)

        # zero grad the optimizer
        optimizer.zero_grad()
        # do the forward step of model
        outputs = model(inputs)
        # calculate loss
        loss = nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))
        # backward step the loss
        loss.backward()
        # step optimizer
        optimizer.step()

        # accumulate running loss
        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)

        # update progress bar
        progress_bar.set_postfix({
            "Batch": batch_idx + 1,
            "Loss": avg_loss,
            "Images": (batch_idx + 1) * data_loader.batch_size
        })

        # if you have a scheduler, you either need to
        # step it here or you have to step it after
        # the epoch. here, we are not using any learning
        # rate scheduler

    print(f"Training Loss: {avg_loss:.4f}")


def evaluate(data_loader, model, device):
    """
    This function does evaluation for one epoch
    :param data_loader: this is the pytorch dataloader
    :param model: pytorch model
    :param device: cuda/cpu
    """

    # put model in evaluation mode
    model.eval()

    # init lists to store targets and outputs
    final_targets = []
    final_outputs = []

    # we use no_grad context
    with torch.no_grad():
        progress_bar = tqdm(data_loader, total=len(data_loader), desc="Evaluating")
        for data in progress_bar:
            inputs = data["image"]
            targets = data["targets"]
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)

            # do the forward step to generate prediction
            output = model(inputs)

            # convert targets and outputs to lists
            targets = targets.detach().cpu().numpy().tolist()
            output = output.detach().cpu().numpy().tolist()

            # extend the original list
            final_targets.extend(targets)
            final_outputs.extend(output)

    # return final output and final targets
    return final_outputs, final_targets