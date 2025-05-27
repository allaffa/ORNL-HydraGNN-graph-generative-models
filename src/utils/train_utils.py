import os, random, csv
import tqdm
import torch
from datetime import datetime   

from torch_geometric.data import Data, Batch
from typing import Dict, Tuple, Any, List, Optional

import hydragnn

from src.processes.diffusion import DiffusionProcess
from src.processes.equivariant_diffusion import center_gravity
from src.utils.logging_utils import ModelLoggerHandler

def per_graph_mse_loss(outputs, targets, data):
    # Initialize the list to hold the per-graph losses
    per_graph_losses = []
    
    # Get the batch vector that indicates which graph each node belongs to
    batch = data.batch
    # Loop through each unique graph in the batch
    for graph_idx in torch.unique(batch):
        # Select the nodes belonging to the current graph
        mask = batch == graph_idx
        
        # Extract the outputs and targets for the current graph
        graph_outputs = outputs[mask]
        graph_targets = targets[mask]
        
        # Compute MSE loss for the current graph
        mse_loss = torch.nn.functional.mse_loss(graph_outputs, graph_targets)
        
        # Append the loss to the list
        per_graph_losses.append(mse_loss)
    # Return the list of per-graph MSE losses
    return torch.mean(torch.tensor(per_graph_losses, requires_grad=True))

def diffusion_loss(outputs, targets, pos_weight=1.0, atom_weight=1.0, predict_x0=False):
    """
    Loss function for graph diffusion models with norm constraint.
    Computes positional loss with norm constraint and atom type loss.
    
    Args:
        outputs: Model outputs [atom_predictions, position_predictions]
        targets: Target values [atom_targets, position_targets]
        pos_weight: Weight multiplier for the position loss
        atom_weight: Weight multiplier for the atom type loss
        predict_x0: If True, model is predicting original positions (x0)
                    If False, model is predicting noise (epsilon)
        
    Returns:
        Tuple of (position_loss, atom_type_loss, metrics_dict)
    """
    # Verify center of gravity constraint on both predictions and targets
    pos_preds = outputs[1]
    pos_targets = targets[1]

    # Double-check center of gravity (should be close to zero for both)
    # Calculate center of mass for the predictions and targets
    pred_center = torch.mean(pos_preds, dim=0)
    target_center = torch.mean(pos_targets, dim=0)
    
    # For x0 prediction, we need additional components to prevent collapse to zero
    if predict_x0:
        # Center of gravity penalty - weaker in x0-prediction mode as it's less critical
        cog_penalty_weight = 0.0  
        cog_penalty = cog_penalty_weight * (torch.norm(pred_center)**2)
        
        # Basic MSE loss for positions
        pos_mse_loss = torch.nn.functional.mse_loss(pos_preds, pos_targets)
        
        # Add norm constraint to ensure predictions have appropriate scale
        # Calculate norms for each sample and across the batch
        pred_norm = torch.norm(pos_preds, dim=1).mean()
        target_norm = torch.norm(pos_targets, dim=1).mean()
        
        # Anti-collapse term: strongly penalize when predictions are too small
        # This specifically addresses the zero-collapse problem
        # Uses a one-sided penalty that only activates when predictions are smaller than targets
        anti_collapse_weight = 0.0
        anti_collapse_loss = torch.relu(target_norm - pred_norm)  # Only penalize if pred_norm < target_norm
        
        # Encourage prediction variance to match target variance
        # This helps prevent mode collapse to a single point
        pred_var = torch.var(pos_preds, dim=0).mean()
        target_var = torch.var(pos_targets, dim=0).mean()
        variance_match_weight = 0.0
        variance_match_loss = torch.abs(pred_var - target_var)
        
        # Combined position loss with increased weight
        pos_loss = pos_weight * (
            pos_mse_loss + 
            anti_collapse_weight * anti_collapse_loss + 
            variance_match_weight * variance_match_loss +
            cog_penalty
        )
        
        # Track these additional metrics
        metrics_extra = {
            'anti_collapse_loss': anti_collapse_loss.item(),
            'variance_match_loss': variance_match_loss.item(),
            'pred_var': pred_var.item(),
            'target_var': target_var.item(),
        }
    else:
        # For epsilon prediction, use the original approach
        cog_penalty_weight = 0.0
        cog_penalty = cog_penalty_weight * (torch.norm(pred_center)**2)
        pos_mse_loss = torch.nn.functional.mse_loss(pos_preds, pos_targets)
        #print(f'MSE loss: {pos_mse_loss}')
        
        pred_norm = torch.norm(pos_preds, dim=1).mean()
        target_norm = torch.norm(pos_targets, dim=1).mean()

        #print(f'norms (pred, target): {pred_norm},{target_norm}')
        
        norm_constraint_weight = 0.0
        norm_constraint_loss = torch.abs(pred_norm - target_norm)
        
        pos_loss = pos_weight * (pos_mse_loss + norm_constraint_weight * norm_constraint_loss + cog_penalty)
        
        # Empty extra metrics for this mode
        metrics_extra = {}
    
    # Cross entropy loss for atom types (with reduced weight)
    atom_loss = atom_weight * torch.nn.functional.cross_entropy(outputs[0], targets[0])
    
    # Create a dictionary of metrics to track
    metrics = {
        'pos_mse_loss': pos_mse_loss.item(),
        'cog_penalty': cog_penalty.item(),
        'pred_center_norm': torch.norm(pred_center).item(),
        'target_center_norm': torch.norm(target_center).item(),
        'pred_norm': pred_norm.item(),
        'target_norm': target_norm.item(),
        'weighted_pos_loss': pos_loss.item(),
        'weighted_atom_loss': atom_loss.item(),
        'prediction_mode': 'x0' if predict_x0 else 'epsilon'
    }
    
    # Add any mode-specific metrics
    metrics.update(metrics_extra)
    
    return pos_loss, atom_loss, metrics

def get_device():
    """
    Determine the appropriate device (MPS, CUDA, or CPU) for training.

    Returns:
    --------
        device: The torch device to use for training
    """

    return hydragnn.utils.distributed.get_device()

def postprocess_model_outputs(outputs, data, predict_x0=False):   
    """
    Process the model outputs - either predict noise (epsilon) or original positions (x0)
    Properly handles batched data by applying operations per-molecule rather than across the batch.
    
    Args:
        outputs: Raw model outputs [atom_predictions, position_predictions]
        data: The input data batch
        predict_x0: If True, the model is predicting x0 (original positions)
                    If False, the model is predicting noise (epsilon)
    
    Returns:
        Processed outputs
    """
    # Apply center of gravity operation per molecule, not across the entire batch
    pos_preds = outputs[1] - data.pos  # Shape [num_nodes_total, 3]
    
    # Check if we have batch information to separate molecules
    if hasattr(data, 'batch') and data.batch is not None:
        # Process each molecule separately and then recombine
        batch_idx = data.batch
        unique_batch_indices = torch.unique(batch_idx)
        for mol_idx in unique_batch_indices:
            # Get molecule mask
            mol_mask = (batch_idx == mol_idx)
            # Apply center_gravity to just this molecule's positions
            mol_positions = pos_preds[mol_mask]
            centered_positions = center_gravity(mol_positions)
            # Update the original tensor
            pos_preds[mol_mask] = centered_positions
            
        outputs[1] = pos_preds
    else:
        # No batch information, assume single molecule
        outputs[1] = center_gravity(pos_preds)
    
    return outputs

def train_epoch(model, loss_fun, optimizer, dataloader, device, logger_handler, epoch, transform_fn=None):
    """
    Train the model for one epoch.

    Args:
    -----
        model (torch.nn.Module): The model to be trained
        loss_fun (callable): The loss function to compute the training loss
        optimizer (torch.optim.Optimizer): The optimizer for updating weights
        dataloader (torch.utils.data.DataLoader): The training data loader
        device (torch.device): The device to run training on
        logger_handler (ModelLoggerHandler): Logger for tracking metrics
        epoch (int): Current epoch number
        transform_fn (callable, optional): Transform function to apply to each batch

    Returns:
    --------
        tuple: (epoch_loss, epoch_pos_loss, epoch_atom_loss, batch_count)
    """
    comm_size, comm_rank = hydragnn.utils.distributed.get_comm_size_and_rank()

    model.train()
    epoch_loss = 0
    epoch_pos_loss = 0  # Positional loss (MSE)
    epoch_atom_loss = 0  # Atom type loss (Cross-entropy)
    batch_count = 0

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        # Apply transform to add fresh noise if transform function is provided
        if transform_fn is not None:
            # For batches we need to transform each sample individually
            transformed_samples = [transform_fn(data.clone()) for data in batch.to_data_list()]
            batch = Batch.from_data_list(transformed_samples)

        # Use global prediction mode from transform function
        predict_x0 = getattr(transform_fn, 'predict_x0', False) if transform_fn else False
        # Forward pass
        batch = batch.to(device)
        outputs = model(batch)

        # outputs = postprocess_model_outputs(model(batch), batch, predict_x0=predict_x0)
        # then, compute the loss
        loss_pos, loss_atom, loss_metrics = loss_fun(
            outputs, 
            (batch.y[:, :5], batch.y[:, 5:]),
            predict_x0=predict_x0
        )

        # get them another way:
        loss = loss_pos+loss_atom
        # loss_atom = torch.tensor(0)
        optimizer.step()

        # Backward pass and optimization
        loss.backward()
        # Apply gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate losses
        epoch_loss += loss.item()
        epoch_pos_loss += loss_pos.item()
        epoch_atom_loss += loss_atom.item()
        batch_count += 1
        
        # Calculate norms to monitor predicted vs actual noise scale
        with torch.no_grad():
            pred_noise_norm = torch.norm(outputs[1], dim=1).mean().item()
            actual_noise_norm = torch.norm(batch.y[:, 5:], dim=1).mean().item()
            noise_scale_ratio = pred_noise_norm / (actual_noise_norm + 1e-6)  # avoid division by zero
        
        # Log batch metrics including separate loss components and noise norms
        loss_components = {
            "position_loss": loss_pos, 
            "atom_type_loss": loss_atom,
            "pred_noise_norm": pred_noise_norm,
            "actual_noise_norm": actual_noise_norm,
            "noise_scale_ratio": noise_scale_ratio,
        }
        
        # Add current time information to batch logging if available
        if hasattr(batch, 'time_norm') and torch.is_tensor(batch.time_norm):
            # Use the mean time across the batch for simplicity
            avg_time = batch.time_norm.float().mean().item()
            loss_components["time"] = avg_time
    
        if 0 == comm_rank:
            logger_handler.log_batch(
                loss, batch_count, epoch, len(dataloader), "train", loss_components
            )
    
    return epoch_loss, epoch_pos_loss, epoch_atom_loss, batch_count

def validate_epoch(model, loss_fun, dataloader, device, logger_handler, epoch, transform_fn=None):
    """
    Validate the model for one epoch.

    Args:
    -----
        model (torch.nn.Module): The model to be validated
        loss_fun (callable): The loss function to compute the validation loss
        dataloader (torch.utils.data.DataLoader): The validation data loader
        device (torch.device): The device to run validation on
        logger_handler (ModelLoggerHandler): Logger for tracking metrics
        epoch (int): Current epoch number
        transform_fn (callable, optional): Transform function to apply to each batch

    Returns:
    --------
        tuple: (epoch_val_loss, epoch_val_pos_loss, epoch_val_atom_loss, batch_count_val)
    """
    comm_size, comm_rank = hydragnn.utils.distributed.get_comm_size_and_rank()

    model.eval()
    epoch_val_loss = 0
    epoch_val_pos_loss = 0
    epoch_val_atom_loss = 0
    batch_count_val = 0
    
    for val_batch_idx, batch in enumerate(dataloader):
        with torch.no_grad():
            # Apply transform to add fresh noise if transform function is provided
            if transform_fn is not None:
                # For batches we need to transform each sample individually
                transformed_samples = [transform_fn(data.clone()) for data in batch.to_data_list()]
                batch = Batch.from_data_list(transformed_samples)
            
            # Use global prediction mode from transform function
            predict_x0 = getattr(transform_fn, 'predict_x0', False) if transform_fn else False
            
            # Forward pass
            batch = batch.to(device)
            # outputs = postprocess_model_outputs(model(batch), batch, predict_x0=predict_x0)
            outputs = model(batch)
            loss_pos, loss_atom, loss_metrics = loss_fun(
                outputs, 
                [batch.y[:, :5], batch.y[:, 5:]], 
                predict_x0=predict_x0
            )
            loss = loss_pos + loss_atom  # Combine the weighted losses

            # Accumulate losses
            epoch_val_loss += loss.item()
            epoch_val_pos_loss += loss_pos.item()
            epoch_val_atom_loss += loss_atom.item()
            batch_count_val += 1

            # Calculate norms to monitor predicted vs actual noise scale
            pred_noise_norm = torch.norm(outputs[1], dim=1).mean().item()
            actual_noise_norm = torch.norm(batch.y[:, 5:], dim=1).mean().item()
            noise_scale_ratio = pred_noise_norm / (actual_noise_norm + 1e-6)  # avoid division by zero
            
            # Log batch metrics including loss components and noise norms
            val_loss_components = {
                "position_loss": loss_pos,
                "atom_type_loss": loss_atom,
                "pred_noise_norm": pred_noise_norm,
                "actual_noise_norm": actual_noise_norm,
                "noise_scale_ratio": noise_scale_ratio,
                # "norm_constraint_loss": loss_metrics['norm_constraint_loss']
            }
            
            # Add current time information to batch logging if available
            if hasattr(batch, 'time_norm') and torch.is_tensor(batch.time_norm):
                # Use the mean time across the batch for simplicity
                avg_time = batch.time_norm.float().mean().item()
                val_loss_components["time"] = avg_time
            
            if 0 == comm_rank:
                logger_handler.log_batch(
                    loss,
                    batch_count_val,
                    epoch,
                    len(dataloader),
                    "val",
                    val_loss_components,
                )

    return epoch_val_loss, epoch_val_pos_loss, epoch_val_atom_loss, batch_count_val

def log_epoch_metrics(logger_handler, epoch, train_losses, val_losses, optimizer):
    """
    Calculate and log metrics for the epoch.

    Args:
    -----
        logger_handler (ModelLoggerHandler): Logger for tracking metrics
        epoch (int): Current epoch number
        train_losses (tuple): (epoch_loss, epoch_pos_loss, epoch_atom_loss, batch_count)
        val_losses (tuple): (epoch_val_loss, epoch_val_pos_loss, epoch_val_atom_loss, batch_count_val)
        optimizer (torch.optim.Optimizer): The optimizer used for training

    Returns:
    --------
        float: The average epoch loss
    """

    comm_size, comm_rank = hydragnn.utils.distributed.get_comm_size_and_rank()

    epoch_loss, epoch_pos_loss, epoch_atom_loss, batch_count = train_losses
    (
        epoch_val_loss,
        epoch_val_pos_loss,
        epoch_val_atom_loss,
        batch_count_val,
    ) = val_losses

    # Calculate average losses
    avg_epoch_val_loss = epoch_val_loss / batch_count_val
    avg_epoch_loss = epoch_loss / batch_count

    # Calculate average loss components
    avg_pos_loss = epoch_pos_loss / batch_count
    avg_atom_loss = epoch_atom_loss / batch_count
    avg_val_pos_loss = epoch_val_pos_loss / batch_count_val
    avg_val_atom_loss = epoch_val_atom_loss / batch_count_val

    # Create loss component averages dictionary
    loss_component_avgs = {
        "train_position_loss": avg_pos_loss,
        "train_atom_type_loss": avg_atom_loss,
        "val_position_loss": avg_val_pos_loss,
        "val_atom_type_loss": avg_val_atom_loss,
    }

    # Log epoch metrics with loss component averages
    if 0 == comm_rank:
        logger_handler.log_epoch(
            epoch,
            avg_epoch_loss,
            avg_epoch_val_loss,
            optimizer.param_groups[0]["lr"],
            loss_component_avgs,
        )

    return avg_epoch_loss

def train_model(
    model,
    loss_fun,
    optimizer,
    train_dataloader,
    val_dataloader,
    num_epochs,
    logger=None,
    config=None,
    save_freq=1,
    save_best=True,
    model_name="model",
    scheduler=None,
    train_transform=None
):
    """
    Trains a given model using the provided loss function, optimizer, and data loader.

    Args:
    -----
        model (torch.nn.Module): The model to be trained.
        loss_fun (callable): The loss function to compute the training loss.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model's weights.
        train_dataloader (torch.utils.data.DataLoader): An iterable over the training dataset.
        val_dataloader (torch.utils.data.DataLoader): An iterable over the validation dataset.
        num_epochs (int): The number of epochs for training.
        logger: The logger object to use for experiment tracking.
        config (dict, optional): Configuration dictionary for logger initialization.
        save_freq (int): Frequency of saving model checkpoints (in epochs).
        save_best (bool): Whether to save the best model based on loss.
        model_name (str): Base name for saved model files.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
        train_transform (callable, optional): Transform function to apply to each batch.

    Returns:
    --------
        model: The trained model
    """
    comm_size, comm_rank = hydragnn.utils.distributed.get_comm_size_and_rank()

    device = get_device()
    model.to(device)

    logger_handler = None

    # Initialize logger handler
    if 0 == comm_rank:
        logger_handler = ModelLoggerHandler(
            logger=logger, model_name=model_name, save_freq=save_freq, save_best=save_best
        )
        logger_handler.setup(model, config)
    
    for epoch in tqdm.tqdm(range(num_epochs)):
        # Train for one epoch
        train_losses = train_epoch(
            model, loss_fun, optimizer, train_dataloader, device, logger_handler, epoch, train_transform
        )

        # Validate the model
        val_losses = validate_epoch(
            model, loss_fun, val_dataloader, device, logger_handler, epoch, train_transform
        )

        # Log the epoch metrics
        avg_epoch_loss = log_epoch_metrics(
            logger_handler, epoch, train_losses, val_losses, optimizer
        )
        
        # Update learning rate with scheduler if provided
        if scheduler is not None:
            # For ReduceLROnPlateau, step needs the validation loss
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_losses[0] / val_losses[3])  # avg_val_loss
            else:
                scheduler.step()
                
        # Handle end of epoch (saving checkpoints)
        if 0 == comm_rank:
            logger_handler.handle_epoch_end(epoch, model, optimizer, avg_epoch_loss)

    print("Training complete!")
    if 0 == comm_rank:
        logger_handler.finish()
    return model

def get_train_transform(dp: DiffusionProcess, predict_x0=False):
    """
    Returns a training transform function for the QM9 dataset. Encompasses
    the forward noising process, time insertion, and formatting for the
    denoising model.

    Args:
    -----
    dp (DiffusionProcess):
        The diffusion process object.
    predict_x0 (bool):
        If True, the model predicts the original positions (x0) instead of noise.
    head_types (list):
        The types of heads in the denoising model.
    out_indices (list):
        The output indices of the denoising model.
    graph_feature_dim (list):
        The dimensions of the graph features.
    node_feature_dim (list):
        The dimensions of the node features.
    """

    def train_transform(data: Data):
        data.t = 0  # default

        # Only use atom type features
        data.x = data.x[:, :5].float()
        
        # Save the original positions for x0 prediction case
        original_pos = data.pos.clone()

        # randomly sample a t
        t = random.randint(0, dp.timesteps - 1)
        # noise the data 
        noised_data = dp.forward_process_sample(
            data, t
        )  
        # the noised data object has the following attributes:
        # - x: noised atom types 
        # - x_t: probability distribution of atom types 
        # - pos: noised positions (eg z_t^(x) from Hoogeboom et al.)
        # - ypos: epsilon for the positions, with center of gravity removed
        # - t: the time step t
        # - pos_mu: set to None
        # - pos_sigma: set to None
        
        # Store the timestep t in the data object so we can access it for logging
        # Store as tensors so they will be properly batched
        noised_data.time_step = torch.tensor([t], device=data.x.device)
        # Also store normalized time (t/timesteps) for binning
        noised_data.time_norm = torch.tensor([t / (dp.timesteps - 1.0)], device=data.x.device)

        # insert time t into the node features
        noised_data, time_vec = insert_t(noised_data, t, dp.timesteps)

        # set y_loc for hydragnn reasons
        # noised_data.y_loc = torch.tensor(
        #     [0, 5, 8], dtype=torch.int64, device=data.x.device
        # ).unsqueeze(0)

        # Store the diffusion alpha for the current timestep (needed for x0 prediction)
        noised_data.alpha_t = dp.alphas[t]

        if predict_x0:
            # For x0 prediction, the target is the original position
            noised_data.ypos_original = noised_data.ypos.clone()  # Save the original noise for debugging
            noised_data.y = torch.hstack([data.x, center_gravity(original_pos)])
        else:
            # For epsilon prediction, the target is the noise (traditional approach)
            noised_data.y = torch.hstack([data.x, noised_data.ypos])
        return noised_data

    # Store predict_x0 as an attribute on the function for access in train_epoch
    train_transform.predict_x0 = predict_x0
    
    return train_transform

def get_hydra_transform():
    """
    Returns a training transform function for the QM9 dataset. Encompasses
    the forward noising process, time insertion, and formatting for the
    denoising model.

    Args:
    -----
    dp (DiffusionProcess):
        The diffusion process object.
    head_types (list):
        The types of heads in the denoising model.
    out_indices (list):
        The output indices of the denoising model.
    graph_feature_dim (list):
        The dimensions of the graph features.
    node_feature_dim (list):
        The dimensions of the node features.
    """
    def hydra_transform(data: Data):
        data.y_loc = torch.tensor(
            [0, 5, 8], dtype=torch.int64, device=data.x.device
        ).unsqueeze(0)
        return data

    return hydra_transform


def get_deterministic_transform(timesteps: int, predict_x0=False):
    """
    Returns a training transform function that applies a deterministic
    deformation to atom positions instead of random noise.
    This helps diagnose position learning issues by creating a simple,
    deterministic target.

    Args:
    -----
    timesteps (int):
        Number of diffusion timesteps to simulate
    predict_x0 (bool):
        If True, the model predicts the original positions (x0) instead of noise.
    """

    def deterministic_deformation(positions, time_factor=1.0):
        """
        Apply a deterministic, structured deformation to positions.
        The deformation maintains zero center of gravity.
        
        Args:
            positions: Tensor of atom positions [num_atoms, 3]
            time_factor: Factor to scale the deformation (0.0 to 1.0)
            
        Returns:
            deformed_positions: Tensor of deformed positions
            deformation: The deformation field applied (for training target)
        """
        # Calculate distance from origin for each atom
        distances = torch.norm(positions, dim=1, keepdim=True)
        
        # Create a radial deformation field (expand/contract based on distance)
        # Scale intensity based on time_factor
        intensity = 0.3 * time_factor
        
        # Direction vectors for each atom (normalized positions)
        directions = positions / (distances + 1e-8)
        
        # Apply the deformation - intensity varies with distance (sine wave pattern)
        deformation = directions * (torch.sin(distances * 3.14159) * intensity)
        
        # Ensure deformation has zero center of gravity
        deformation = center_gravity(deformation)
        
        # Apply deformation
        deformed_positions = positions + deformation
        
        return deformed_positions, deformation

    def train_transform(data: Data):
        data.t = 0  # default

        # Only use atom type features
        data.x = data.x[:, :5].float()
        
        # Save the original positions for x0 prediction case
        original_pos = data.pos.clone()

        # randomly sample a t
        t = random.randint(0, timesteps - 1)
        time_factor = t / (timesteps - 1.0)
        
        # Create a copy of the data
        noised_data = data.clone().detach_()
        # noised_data = data.clone()
        
        # Apply deterministic deformation
        deformed_positions, deformation = deterministic_deformation(
            original_pos, time_factor
        )
        
        # Set the transformed properties
        noised_data.pos = deformed_positions
        noised_data.ypos = deformation
        noised_data.t = t
        
        # Store the timestep t in the data object so we can access it for logging
        # Store as tensors so they will be properly batched
        noised_data.time_step = torch.tensor([t], device=data.x.device)
        # Also store normalized time (t/timesteps) for binning
        noised_data.time_norm = torch.tensor([time_factor], device=data.x.device)

        # insert time t into the node features
        noised_data, time_vec = insert_t(noised_data, t, timesteps)

        # set y_loc for hydragnn reasons
        noised_data.y_loc = torch.tensor(
            [0, 5, 8], dtype=torch.int64, device=data.x.device
        ).unsqueeze(0)

        # Create alphas similar to diffusion process
        alphas = torch.linspace(1.0, 0.1, timesteps)
        noised_data.alpha_t = alphas[t]

        if predict_x0:
            # For x0 prediction, the target is the original position
            noised_data.ypos_original = deformation.clone()  # Save the original deformation for debugging
            noised_data.y = torch.hstack([data.x, center_gravity(original_pos)])
        else:
            # For epsilon prediction, the target is the deformation
            noised_data.y = torch.hstack([data.x, deformation])
        
        return noised_data

    # Store attributes on the function for access in train_epoch
    train_transform.predict_x0 = predict_x0
    train_transform.deterministic = True
    
    return train_transform


def insert_t(data: Data, t: int, T: int):
    """
    Insert time t into the node features of the data before passing
    to the denoising model.

    Args:
    -----
    data (Data):
        Noised sample from the diffusion process.
    t (int):
        The time step to insert into the node features.
    T (int):
        The total number of time steps in the diffusion process.
    """
    data_ins = data.clone().detach_()
    # concatenate node features and (scaled) time feature
    time_vec = (
        torch.ones((data_ins.num_nodes, 1), device=data_ins.x.device) * t / (T - 1.0)
    )
    data_ins.x = torch.hstack([data_ins.x, time_vec])  # (n_nodes, 6)
    return data_ins, time_vec
