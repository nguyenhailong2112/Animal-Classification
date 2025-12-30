import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from myModel import CNN
from myDataset import AnimalDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Resize, Compose, Normalize, RandomHorizontalFlip,ColorJitter
from torch.optim import SGD
from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, precision_score, recall_score
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description="Animal Classification")

    parser.add_argument("--data-path", "-d", type=str, default="myAnimalDataset/seen_classes", help="path to dataset")
    parser.add_argument("--log-path", "-o", type=str, default="my_tensorboard", help="Tensorboard log directory")
    parser.add_argument("--checkpoint-path", "-c", type=str, default="my_models", help="Model checkpoint directory")
    parser.add_argument("--image-size", "-i", type=int, default=224, help="Input image size")
    parser.add_argument("--batch-size", "-b", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning-rate", "-l", type=float, default=0.001, help="Learning rate of optimizer")
    parser.add_argument("--momentum", "-m", type=float, default=0.9, help="Momentum for optimizer")
    parser.add_argument("--k-folds", "-k", type=int, default=4, help="Number of folds for K-Fold Cross Validation")
    parser.add_argument("--resume", "-r", type=str, help="Path to checkpoint to resume training")

    args = parser.parse_args()
    return args

def plot_confusion_matrix(writer, cm, class_names, epoch, fold):
    figure = plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap="hsv")
    plt.title(f"Confusion Matrix (Fold {fold})")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    threshold = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    writer.add_figure(f"confusion_matrix/fold_{fold}", figure, epoch)

def train(args):
    transform = Compose([
        Resize((args.image_size, args.image_size)),
        RandomHorizontalFlip(),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = AnimalDataset(root=args.data_path, train=True, transform=transform)
    test_dataset = AnimalDataset(root=args.data_path, train=False, transform=transform)

    all_labels = [label for _, label in full_dataset]
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    fold_results = []

    print(f"\nStarting {args.k_folds}-Fold Cross Validation")
    print(f"\nTổng số ảnh bộ Train: {len(full_dataset)}")

    start_fold = 0
    start_epoch = 0
    best_global_loss = float('inf')
    best_global_acc = 0
    best_fold_id = 0

    if args.resume:
        checkpoint = torch.load(args.resume)
        start_fold = checkpoint['fold']
        start_epoch = checkpoint['epoch'] + 1
        best_global_loss = checkpoint['best_global_loss']
        best_global_acc = checkpoint['best_global_acc']
        best_fold_id = checkpoint['best_fold_id']
        print(f"\nResuming from Fold {start_fold + 1}, Epoch {start_epoch}")
        print(f"Best Global Loss: {best_global_loss:.4f}, Best Global Acc: {best_global_acc:.4f}")

    all_indices = np.arange(len(full_dataset))
    for fold, (train_ids, val_ids) in enumerate(kfold.split(all_indices)):
        if fold < start_fold:
            continue

        print(f"FOLD {fold + 1}/{args.k_folds}")

        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)

        train_loader = DataLoader(
            full_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=2,
            pin_memory=True
        )

        val_loader = DataLoader(
            full_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=2,
            pin_memory=True
        )

        model = CNN(num_classes=len(full_dataset.categories))
        model.to(device)

        optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

        fold_writer = SummaryWriter(log_dir=os.path.join(args.log_path, f"fold_{fold + 1}"))

        best_val_loss = float('inf')
        best_fold_acc = 0.0
        no_improve = 0
        best_fold_f1 = 0.0
        best_fold_precision = 0.0
        best_fold_recall = 0.0

        if args.resume and fold == start_fold:
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            best_fold_acc = checkpoint['best_fold_acc']
            best_val_loss = checkpoint['best_val_loss']
            no_improve = checkpoint.get('no_improve', 0)
            best_fold_f1 = checkpoint.get('best_fold_f1', 0.0)
            best_fold_precision = checkpoint.get('best_fold_precision', 0.0)
            best_fold_recall = checkpoint.get('best_fold_recall', 0.0)

        for epoch in range(start_epoch, args.epochs):
            # Training
            model.train()
            epoch_loss = []
            progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch", colour="cyan")

            for batch_idx, (images, labels) in enumerate(progress_bar_train):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                epoch_loss.append(loss.item())

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                fold_writer.add_scalar("Train/Loss", loss.item(), global_step=epoch * len(train_loader) + batch_idx)

            # Validation
            model.eval()
            val_preds = []
            val_labels = []
            val_loss = []
            progress_bar_val = tqdm(val_loader, desc="Validation", unit="batch", colour="yellow")
            with torch.no_grad():
                for images, labels in progress_bar_val:
                    images = images.to(device)
                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
                    val_loss.append(criterion(outputs, labels.to(device)).item())

            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='macro')
            val_precision = precision_score(val_labels, val_preds, average='macro')
            val_recall = recall_score(val_labels, val_preds, average='macro')

            avg_val_loss = np.mean(val_loss)
            print(f"[Fold {fold + 1}] Epoch {epoch + 1}: Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.4f}")

            fold_writer.add_scalar("Val/Loss", avg_val_loss, global_step=epoch)
            fold_writer.add_scalar("Val/Accuracy", val_acc, global_step=epoch)
            plot_confusion_matrix(fold_writer, confusion_matrix(val_labels, val_preds), full_dataset.categories, epoch, fold + 1)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_fold_acc = val_acc
                best_fold_f1 = val_f1
                best_fold_precision = val_precision
                best_fold_recall = val_recall
                no_improve = 0
                torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f"best_fold_{fold + 1}.pth"))

                if avg_val_loss < best_global_loss:
                    best_global_loss = avg_val_loss
                    best_fold_id = fold + 1
                    torch.save(model.state_dict(), os.path.join(args.checkpoint_path, "bestFold.pth"))
            else:
                no_improve += 1
                if no_improve >= 5:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            fold_writer.add_scalar("Val/F1", val_f1, epoch)
            fold_writer.add_scalar("Val/Precision", val_precision, epoch)
            fold_writer.add_scalar("Val/Recall", val_recall, epoch)

            checkpoint = {
                'fold': fold,
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_global_loss': best_global_loss,
                'best_global_acc': best_global_acc,
                'best_fold_id': best_fold_id,
                'best_val_loss': best_val_loss,
                'best_fold_acc': best_fold_acc,
                'best_fold_f1': best_fold_f1,
                'best_fold_precision': best_fold_precision,
                'best_fold_recall': best_fold_recall
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_path, f"checkpoint_fold{fold + 1}_epoch{epoch + 1}.pth"))

        start_epoch = 0

        fold_results.append({
            'loss': best_val_loss,
            'accuracy': best_fold_acc,
            'f1': best_fold_f1,
            'precision': best_fold_precision,
            'recall': best_fold_recall
        })

    print("\nK-FOLD METRICS")

    for fold, metrics in enumerate(fold_results):
        print(f"Fold {fold + 1}:")
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Fold {fold + 1} completed!\n")

    mean_loss = np.mean([m['loss'] for m in fold_results])
    std_loss = np.std([m['loss'] for m in fold_results])
    print(f"\nMean Loss: {mean_loss:.4f}")
    print(f"Std Loss: {std_loss:.4f}")

    mean_accuracy = np.mean([m['accuracy'] for m in fold_results])
    std_accuracy = np.std([m['accuracy'] for m in fold_results])
    print(f"\nMean Accuracy: {mean_accuracy:.4f}")
    print(f"Std Deviation: {std_accuracy:.4f}")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    best_model = CNN(num_classes=len(full_dataset.categories)).to(device)
    best_model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, "bestFold.pth")))

    def evaluate(eval_model, data_loader, criterion):
        eval_model.eval()
        eval_labels = []
        all_preds = []
        total_loss = []
        with torch.no_grad():
            for eval_images, eval_labels_batch in data_loader:
                eval_images = eval_images.to(device)
                eval_labels_batch = eval_labels_batch.to(device)
                outputs = eval_model(eval_images)
                loss = criterion(outputs, eval_labels_batch)
                total_loss.append(loss.item())
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                eval_labels.extend(eval_labels_batch.cpu().numpy())
        return {
            'loss': np.mean(total_loss),
            'accuracy': accuracy_score(eval_labels, all_preds),
            'f1': f1_score(eval_labels, all_preds, average='macro'),
            'precision': precision_score(eval_labels, all_preds, average='macro'),
            'recall': recall_score(eval_labels, all_preds, average='macro')
        }

    test_metrics = evaluate(best_model, test_loader, criterion)
    print("\nTest Set Metrics")
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1-Score: {test_metrics['f1']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")

if __name__ == '__main__':
    args = get_args()
    train(args)