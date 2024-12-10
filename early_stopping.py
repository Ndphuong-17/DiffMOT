import torch
import os.path as osp

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model, epoch, optimizer, scheduler, model_dir, dataset):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, epoch, optimizer, scheduler, model_dir, dataset)
        elif score > (self.best_score) and score <= 0.1:
            self.counter += 1
            print(f"Validation loss did not improve. Counter: {self.counter}/{self.patience}")
            self.save_checkpoint(model, epoch, optimizer, scheduler, model_dir, dataset)
            if self.counter >= self.patience:
                self.early_stop = True
        elif score < (self.best_score) and score > 0.1:
            self.counter += 1
            print(f"Validation IoU did not improve. Counter: {self.counter}/{self.patience}")
            self.save_checkpoint(model, epoch, optimizer, scheduler, model_dir, dataset)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, epoch, optimizer, scheduler, model_dir, dataset)
            self.counter = 0

    def save_checkpoint(self, model, epoch, optimizer, scheduler, model_dir, dataset):
        checkpoint = {
            'ddpm': model.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        checkpoint_path = osp.join(model_dir, f"{dataset}_epoch{epoch}.pt")
        checkpoint_path = osp.normpath(checkpoint_path)  # Normalize the path
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
