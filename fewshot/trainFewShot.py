from fewShotLearning import *
import torch.optim as optim
import seaborn as sns
from torch.cuda.amp import autocast, GradScaler

def get_args():
    parser = argparse.ArgumentParser(description="Few-Shot Learning Training")
    parser.add_argument("--root_dir", default="myAnimalDataset")
    parser.add_argument("--pretrained_path", default="pre_models/best_model.pth")
    parser.add_argument("--checkpoint_dir", default="fewshot_checkpoints")
    parser.add_argument("--log_dir", default="fewshot_logs")
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--query_num", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--resume", help="Checkpoint to resume from")
    return parser.parse_args()

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    train_dataset = FewShotDataset(
        root_dir=args.root_dir,
        mode='train',
        n_way=args.n_way,
        k_shot=args.k_shot,
        query_num=args.query_num
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        pin_memory=True,
        collate_fn=lambda x: {
            'support': torch.stack([ep['support'] for ep in x]),
            'query': torch.stack([ep['query'] for ep in x]),
            'support_labels': torch.stack([ep['support_labels'] for ep in x]),
            'query_labels': torch.stack([ep['query_labels'] for ep in x])
        }
    )

    val_dataset = FewShotDataset(
        root_dir=args.root_dir,
        mode='val',
        n_way=args.n_way,
        k_shot=args.k_shot,
        query_num=args.query_num
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda x: {
            'support': torch.stack([ep['support'] for ep in x]),
            'query': torch.stack([ep['query'] for ep in x]),
            'support_labels': torch.stack([ep['support_labels'] for ep in x]),
            'query_labels': torch.stack([ep['query_labels'] for ep in x])
        }
    )

    model = FewShotModel(
        pretrained_path=args.pretrained_path,
        n_way=args.n_way,
        k_shot=args.k_shot,
        query_num=args.query_num)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([
        {'params': model.projection.parameters()},
        {'params': model.backbone.layer4.parameters()},
        {'params': model.backbone.fc.parameters()}
    ], lr=args.lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    start_epoch = 0
    best_val_loss = float('inf')
    no_improve = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming from epoch {start_epoch}")

    writer = SummaryWriter(log_dir=args.log_dir)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", colour="cyan")
        for train_batch in progress_bar_train:
            support = train_batch['support'].to(device)
            query = train_batch['query'].to(device)
            labels = train_batch['query_labels'].to(device)

            labels = labels.view(-1)

            optimizer.zero_grad()
            with autocast():
                similarities = model(support, query)
                loss = criterion(similarities, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            preds = torch.argmax(similarities, dim=1)
            correct += (preds == labels.view(-1)).sum().item()
            total += labels.view(-1).size(0)
            progress_bar_train.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        progress_bar_val = tqdm(val_loader, desc="Validation", colour="yellow")
        with torch.no_grad():
            for val_batch in progress_bar_val:
                support = val_batch['support'].to(device)
                query = val_batch['query'].to(device)
                labels = val_batch['query_labels'].to(device)
                similarities = model(support, query)
                loss = criterion(similarities, labels.view(-1))
                val_loss += loss.item()
                preds = torch.argmax(similarities, dim=1)
                val_correct += (preds == labels.view(-1)).sum().item()
                val_total += labels.view(-1).size(0)
                progress_bar_val.set_postfix(val_loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        train_accuracy = correct / total
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        scheduler.step()

        writer.add_scalar('Train/Epoch Loss', avg_loss, epoch)
        writer.add_scalar('Train/Epoch Accuracy', train_accuracy, epoch)
        writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        writer.add_scalar('Val/Accuracy', val_accuracy, epoch)

        print(f"Epoch {epoch + 1}: Loss {avg_loss:.4f}, Acc {train_accuracy:.4f}, Val Loss {avg_val_loss:.4f}, Val Acc {val_accuracy:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'args': vars(args)
            }, os.path.join(args.checkpoint_dir, "best_model.pth"))
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

if __name__ == '__main__':
    args = get_args()
    train(args)