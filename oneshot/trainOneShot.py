import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from oneShotLearning import SiameseDataset
from siameseModel import SiameseNetwork


def get_args():
    parser = argparse.ArgumentParser(description="One-Shot Learning with Siamese Networks")
    parser.add_argument("--root_dir", default="myAnimalDataset")
    parser.add_argument("--pretrained_path", default="pre_models/best_model.pth")
    parser.add_argument("--checkpoint_dir", default="siamese_checkpoints")
    parser.add_argument("--log_dir", default="siamese_logs")
    parser.add_argument("--epochs", type=int, default=21)
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()

def contrastive_loss(distance, label, margin=1.0):
    loss = label * torch.pow(distance, 2) + (1 - label) * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
    return loss.mean()

def calculate_accuracy(distances, labels, threshold=0.5):
    preds = (distances < threshold).astype(np.float32)
    return accuracy_score(labels, preds)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SiameseDataset(
        root_dir=args.root_dir,
        mode='train',
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )

    val_dataset = SiameseDataset(
        root_dir=args.root_dir,
        mode='val',
        transform=transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork(pretrained_path=args.pretrained_path).to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    writer = SummaryWriter(log_dir=args.log_dir)

    start_epoch = 0
    best_val_loss = float('inf')
    if os.path.exists(os.path.join(args.checkpoint_dir, "siamese_best.pth")):
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, "siamese_best.pth"))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Tiếp tục từ epoch {start_epoch}")

    no_improve = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", colour="cyan")
        for image1, image2, label in train_bar:
            image1, image2, label = image1.to(device), image2.to(device), label.to(device)
            optimizer.zero_grad()
            distance = model(image1, image2)
            loss = contrastive_loss(distance, label)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        model.eval()
        total_val_loss = 0.0
        all_distances, all_labels = [], []
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]", colour="yellow")
        with torch.no_grad():
            for image1, image2, label in val_bar:
                image1, image2, label = image1.to(device), image2.to(device), label.to(device)
                distance = model(image1, image2)
                loss = contrastive_loss(distance, label)
                total_val_loss += loss.item()
                all_distances.extend(distance.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                val_bar.set_postfix(val_loss=loss.item())

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = calculate_accuracy(np.array(all_distances), np.array(all_labels))
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, os.path.join(args.checkpoint_dir, "siamese_best.pth"))
            print("Đã lưu best checkpoint!")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    writer.close()

if __name__ == "__main__":
    args = get_args()
    train(args)