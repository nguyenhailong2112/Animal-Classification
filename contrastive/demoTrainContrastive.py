from demoContrastiveLearning import *
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Contrastive Learning - SimCLR Training")
    parser.add_argument("--root_dir", default="myAnimalDataset")
    parser.add_argument("--checkpoint_dir", default="demo_simclr_checkpoints")
    parser.add_argument("--log_dir", default="demo_simclr_logs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--resume", type=str, default="demo_simclr_checkpoints/simclr_best.pth")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    return args


# Training Function
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    transform = get_simclr_transform()

    train_dataset = SimCLRDataset(
        root_dir=args.root_dir,
        mode='train',
        transform=transform
    )

    val_dataset = SimCLRDataset(
        root_dir=args.root_dir,
        mode='val',
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Train dataset size: {len(train_dataset)}, Batches: {len(train_loader)}")
    print(f"Val dataset size: {len(val_dataset)}, Batches: {len(val_loader)}")

    model = SimCLR(projection_dim=args.projection_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    criterion = NTXentLoss(temperature=args.temperature)

    start_epoch = 0
    best_val_loss = float('inf')
    no_improve = 0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from checkpoint at epoch {start_epoch}")

    writer = SummaryWriter(log_dir=args.log_dir)
    accumulation_steps = 16  # Giả lập batch size 64 x 16 = 1024

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_train_loss = 0.0
        step = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", colour="cyan")
        for img1, img2 in train_bar:
            img1, img2 = img1.to(device), img2.to(device)
            if step % accumulation_steps == 0:
                optimizer.zero_grad()
            z_i = model(img1)
            z_j = model(img2)
            loss = criterion(z_i, z_j) / accumulation_steps
            loss.backward()
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
            total_train_loss += loss.item() * accumulation_steps
            train_bar.set_postfix(loss=loss.item() * accumulation_steps)
            step += 1
        avg_train_loss = total_train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        model.eval()
        total_val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]", colour="yellow")
        with torch.no_grad():
            for img1, img2 in val_bar:
                img1, img2 = img1.to(device), img2.to(device)
                z_i = model(img1)
                z_j = model(img2)
                loss = criterion(z_i, z_j)
                total_val_loss += loss.item()
                val_bar.set_postfix(val_loss=loss.item())
        avg_val_loss = total_val_loss / len(val_loader)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")

        scheduler.step(avg_val_loss)

        checkpoint_path = os.path.join(args.checkpoint_dir, f"simclr_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint_path = os.path.join(args.checkpoint_dir, "simclr_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, best_checkpoint_path)
            print(f"Saved best checkpoint with val_loss {best_val_loss:.4f}")
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    args = get_args()
    train(args)