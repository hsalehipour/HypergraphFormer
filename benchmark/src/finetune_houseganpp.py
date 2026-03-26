#!/usr/bin/env python3
"""
Fine-tune HouseGAN++ on Hypergrph1K Dataset.

This script loads the converted Hypergrph1K Dataset (in pickle format) and fine-tunes
the HouseGAN++ model on it.
"""

import argparse
import os
import pickle
import numpy as np
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image

from houseganpp.misc.utils import combine_images, _init_input, selectNodesTypes
from houseganpp.models.models import Discriminator, Generator, compute_gradient_penalty
from benchmark.src.dataset_loader import Hypergraph1kDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune HouseGAN++ on Hypergrph1K Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data parameters
    parser.add_argument("--train_data", type=str, required=True,
                        help="path to training pickle file (e.g., train_data.pkl)")
    parser.add_argument("--val_data", type=str, default=None,
                        help="path to validation pickle file")
    parser.add_argument("--target_set", type=int, default=None,
                        help="filter by number of rooms (None = no filter)")
    
    # Model parameters
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/pretrained.pth",
                        help="path to pretrained generator checkpoint")
    parser.add_argument("--resume", type=str, default=None,
                        help="path to checkpoint to resume from")
    
    # Training parameters
    parser.add_argument("--n_epochs", type=int, default=50,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="size of the batches")
    parser.add_argument("--g_lr", type=float, default=0.0001,
                        help="generator learning rate")
    parser.add_argument("--d_lr", type=float, default=0.0001,
                        help="discriminator learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of second order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=4,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_critic", type=int, default=1,
                        help="number of training steps for discriminator per iter")
    parser.add_argument("--lambda_gp", type=int, default=10,
                        help="lambda for gradient penalty")
    
    # Output parameters
    parser.add_argument("--exp_folder", type=str, default='ft_housegan',
                        help="experiment folder name")
    parser.add_argument("--sample_interval", type=int, default=1000,
                        help="interval between image sampling")
    parser.add_argument("--save_interval", type=int, default=5000,
                        help="interval between saving checkpoints")
    
    return parser.parse_args()


def combine_images_wrapper(samples_batch, nodes_batch, edges_batch, nd_to_sample, ed_to_sample):
    """
    Wrapper for HouseGAN++'s combine_images that handles data format conversion.
    
    Our Hypergrph1K Dataset uses one-hot encoded nodes [num_nodes, 18], but combine_images
    expects scalar node indices. This wrapper converts the data appropriately.
    """
    # Convert one-hot encoded nodes to scalar room type IDs
    # nodes_batch has shape [num_nodes, 18] (one-hot), we need [num_nodes] (scalar IDs)
    # The original RPLAN code does this in test.py line 81: real_nodes = np.where(nds.detach().cpu()==1)[-1]
    nodes_scalar = torch.argmax(nodes_batch, dim=1)  # Get the index of the 1 in each one-hot vector
    
    # Convert nd_to_sample and ed_to_sample to CPU numpy for compatibility
    nd_to_sample_np = nd_to_sample.cpu() if torch.is_tensor(nd_to_sample) else nd_to_sample
    ed_to_sample_np = ed_to_sample.cpu() if torch.is_tensor(ed_to_sample) else ed_to_sample
    
    # Call original combine_images with converted data
    return combine_images(samples_batch, nodes_scalar, edges_batch, nd_to_sample_np, ed_to_sample_np)


def visualize_batch(val_loader, generator, exp_folder, batches_done, device, writer=None):
    """Generate and save sample outputs."""
    generator.eval()
    
    with torch.no_grad():
        try:
            mks, nds, eds, nd_to_sample, ed_to_sample = next(iter(val_loader))
        except StopIteration:
            print("⚠️  Warning: Validation loader is empty")
            return
        except Exception as e:
            print(f"⚠️  Warning: Could not load validation batch: {e}")
            return
        
        batch_size = int(torch.max(nd_to_sample)) + 1
        
        # Move to device
        real_mks = mks.to(device)
        given_nds = nds.to(device)
        given_eds = eds.to(device)
        graph = [given_nds, given_eds]
        
        # Select random nodes
        ind_fixed_nodes, _ = selectNodesTypes(nd_to_sample, batch_size, nds)
        ind_fixed_nodes = ind_fixed_nodes.long()  # Ensure long type for indexing
        
        # Build input
        state = {'masks': real_mks, 'fixed_nodes': ind_fixed_nodes}
        z, given_masks_in, given_nds, given_eds = _init_input(graph, state)
        z = z.to(device)
        given_masks_in = given_masks_in.to(device)
        given_nds = given_nds.to(device)
        given_eds = given_eds.to(device)
        
        # Generate
        gen_mks = generator(z, given_masks_in, given_nds, given_eds)
        
        # Create images (using wrapper to handle one-hot conversion)
        real_imgs_tensor = combine_images_wrapper(real_mks, given_nds, given_eds,
                                                   nd_to_sample, ed_to_sample)
        fake_imgs_tensor = combine_images_wrapper(gen_mks, given_nds, given_eds,
                                                   nd_to_sample, ed_to_sample)
        
        # Save images
        save_image(real_imgs_tensor, f"./exps/{exp_folder}/{batches_done}_real.png",
                   nrow=12, normalize=False)
        save_image(fake_imgs_tensor, f"./exps/{exp_folder}/{batches_done}_fake.png",
                   nrow=12, normalize=False)
        
        # Log to TensorBoard
        if writer is not None:
            try:
                # Ensure 3-channel format for TensorBoard
                if real_imgs_tensor.shape[1] == 1:
                    real_imgs_tensor = real_imgs_tensor.repeat(1, 3, 1, 1)
                    fake_imgs_tensor = fake_imgs_tensor.repeat(1, 3, 1, 1)
                
                writer.add_images('Images/Real', real_imgs_tensor, batches_done)
                writer.add_images('Images/Generated', fake_imgs_tensor, batches_done)
            except Exception as e:
                print(f"⚠️  Warning: Could not log images to TensorBoard: {e}")
    
    generator.train()


def main():
    opt = parse_args()
    
    # Create directories
    os.makedirs(f"./exps/{opt.exp_folder}", exist_ok=True)
    os.makedirs(f"./checkpoints", exist_ok=True)
    
    # Initialize TensorBoard
    tensorboard_dir = f"./runs/{opt.exp_folder}"
    writer = SummaryWriter(tensorboard_dir)
    
    print("=" * 80)
    print("HOUSEGANPP FINE-TUNING ON Hypergrph1K Dataset")
    print("=" * 80)
    print(f"Training data:  {opt.train_data}")
    print(f"Validation data: {opt.val_data if opt.val_data else 'None'}")
    print(f"Pretrained checkpoint: {opt.checkpoint}")
    print(f"Experiment folder: {opt.exp_folder}")
    print(f"TensorBoard logs: {tensorboard_dir}")
    print(f"Batch size: {opt.batch_size}")
    print(f"Epochs: {opt.n_epochs}")
    print(f"Target set (room filter): {opt.target_set if opt.target_set else 'None (all)'}")
    print("=" * 80)
    print()
    
    # Check device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available, training will be slow")
    print()
    
    # Initialize models
    print("Initializing models...")
    generator = Generator()
    discriminator = Discriminator()
    
    # Load pretrained weights or resume
    start_epoch = 0
    batches_done = 0
    
    if opt.resume:
        print(f"Resuming from checkpoint: {opt.resume}")
        checkpoint = torch.load(opt.resume, map_location=device)
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        start_epoch = checkpoint.get('epoch', 0)
        batches_done = checkpoint.get('batches_done', 0)
    elif os.path.exists(opt.checkpoint):
        print(f"Loading pretrained generator: {opt.checkpoint}")
        generator.load_state_dict(torch.load(opt.checkpoint, map_location=device))
    else:
        print(f"⚠️  Warning: No checkpoint found, training from scratch")
    
    generator.to(device)
    discriminator.to(device)
    print("✅ Models initialized")
    print()
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = Hypergraph1kDataset(
        opt.train_data,
        target_set=opt.target_set,
        split='train'
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        collate_fn=collate_fn,
        pin_memory=True
    )
    print(f"✅ Training set: {len(train_dataset)} samples")
    
    val_loader = None
    if opt.val_data and os.path.exists(opt.val_data):
        val_dataset = Hypergraph1kDataset(
            opt.val_data,
            target_set=opt.target_set,
            split='val'
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(8, opt.batch_size),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        print(f"✅ Validation set: {len(val_dataset)} samples")
    print()
    
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.g_lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.d_lr, betas=(opt.b1, opt.b2))
    
    # Loss
    distance_loss = torch.nn.L1Loss()
    
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    # Training loop
    print("Starting training...")
    print("=" * 80)
    
    for epoch in range(start_epoch, opt.n_epochs):
        # Track epoch metrics
        epoch_d_losses = []
        epoch_g_losses = []
        epoch_l1_losses = []
        
        for i, batch in enumerate(train_loader):
            # Unpack batch
            mks, nds, eds, nd_to_sample, ed_to_sample = batch
            
            batch_size = int(torch.max(nd_to_sample)) + 1
            
            # Move to device
            real_mks = mks.to(device)
            given_nds = nds.to(device)
            given_eds = eds.to(device)
            graph = [given_nds, given_eds]
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            for p in discriminator.parameters():
                p.requires_grad = True
            
            optimizer_D.zero_grad()
            
            # Select random nodes
            ind_fixed_nodes, _ = selectNodesTypes(nd_to_sample, batch_size, nds)
            ind_fixed_nodes = ind_fixed_nodes.long()  # Ensure long type for indexing
            
            # Build input
            state = {'masks': real_mks, 'fixed_nodes': ind_fixed_nodes}
            z, given_masks_in, given_nds, given_eds = _init_input(graph, state)
            z = z.to(device)
            given_masks_in = given_masks_in.to(device)
            given_nds = given_nds.to(device)
            given_eds = given_eds.to(device)
            
            # Generate fake masks
            gen_mks = generator(z, given_masks_in, given_nds, given_eds)
            
            # Discriminator loss
            real_validity = discriminator(real_mks, given_nds, given_eds, nd_to_sample)
            fake_validity = discriminator(gen_mks.detach(), given_nds, given_eds, nd_to_sample)
            
            gradient_penalty = compute_gradient_penalty(
                discriminator, real_mks.data, gen_mks.data,
                given_nds.data, given_eds.data, nd_to_sample.data, None
            )
            
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + \
                     opt.lambda_gp * gradient_penalty
            
            d_loss.backward()
            optimizer_D.step()
            
            epoch_d_losses.append(d_loss.item())
            writer.add_scalar('Loss/Discriminator', d_loss.item(), batches_done)
            
            # -----------------
            #  Train Generator
            # -----------------
            if i % opt.n_critic == 0:
                for p in discriminator.parameters():
                    p.requires_grad = False
                
                optimizer_G.zero_grad()
                
                # Generate new samples
                z = Variable(Tensor(np.random.normal(0, 1, (real_mks.shape[0], 128))))
                gen_mks = generator(z, given_masks_in, given_nds, given_eds)
                
                # Generator loss
                fake_validity = discriminator(gen_mks, given_nds, given_eds, nd_to_sample)
                
                # L1 loss for fixed nodes
                if len(ind_fixed_nodes) > 0:
                    l1_loss = distance_loss(
                        gen_mks[ind_fixed_nodes, :, :],
                        given_masks_in[ind_fixed_nodes, 0, :, :]
                    ) * 1000
                else:
                    l1_loss = torch.tensor(0.0).to(device)
                
                g_loss = -torch.mean(fake_validity) + l1_loss
                g_loss.backward()
                optimizer_G.step()
                
                epoch_g_losses.append(g_loss.item())
                epoch_l1_losses.append(l1_loss.item())
                writer.add_scalar('Loss/Generator', g_loss.item(), batches_done)
                writer.add_scalar('Loss/L1', l1_loss.item(), batches_done)
                
                # Print progress
                if batches_done % 100 == 0:
                    print(f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(train_loader)}] "
                          f"[D: {d_loss.item():.4f}] [G: {g_loss.item():.4f}] [L1: {l1_loss.item():.4f}]")
                
                # Save samples
                if batches_done % opt.sample_interval == 0 and batches_done > 0 and val_loader:
                    visualize_batch(val_loader, generator, opt.exp_folder, batches_done, device, writer)
                
                # Save checkpoint
                if batches_done % opt.save_interval == 0 and batches_done > 0:
                    checkpoint_path = f'./checkpoints/{opt.exp_folder}_{batches_done}.pth'
                    torch.save({
                        'epoch': epoch,
                        'batches_done': batches_done,
                        'generator': generator.state_dict(),
                        'discriminator': discriminator.state_dict(),
                        'optimizer_G': optimizer_G.state_dict(),
                        'optimizer_D': optimizer_D.state_dict(),
                    }, checkpoint_path)
                    print(f"💾 Checkpoint saved: {checkpoint_path}")
                
                batches_done += 1
        
        # Epoch summary
        if len(epoch_d_losses) > 0:
            avg_d_loss = np.mean(epoch_d_losses)
            writer.add_scalar('Loss_Epoch/Discriminator', avg_d_loss, epoch)
        
        if len(epoch_g_losses) > 0:
            avg_g_loss = np.mean(epoch_g_losses)
            avg_l1_loss = np.mean(epoch_l1_losses)
            writer.add_scalar('Loss_Epoch/Generator', avg_g_loss, epoch)
            writer.add_scalar('Loss_Epoch/L1', avg_l1_loss, epoch)
            print(f"[Epoch {epoch}] Avg D: {avg_d_loss:.4f}, Avg G: {avg_g_loss:.4f}, Avg L1: {avg_l1_loss:.4f}")
        
        writer.add_scalar('Learning_Rate/Generator', optimizer_G.param_groups[0]['lr'], epoch)
        writer.add_scalar('Learning_Rate/Discriminator', optimizer_D.param_groups[0]['lr'], epoch)
    
    # Save final model
    final_path = f'./checkpoints/{opt.exp_folder}_final.pth'
    torch.save({
        'epoch': opt.n_epochs,
        'batches_done': batches_done,
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
    }, final_path)
    
    writer.close()
    
    print()
    print("=" * 80)
    print(f"✅ Training complete!")
    print(f"📁 Final model: {final_path}")
    print(f"📊 TensorBoard logs: {tensorboard_dir}")
    print(f"   To view: tensorboard --logdir={tensorboard_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
