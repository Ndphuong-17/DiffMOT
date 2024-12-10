import torch
import os.path as osp

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, mode='min'):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

        if mode == 'loss':
            mode = 'min'
        else:
            mode = 'max'

    def __call__(self, val_loss, model, epoch, optimizer, scheduler, model_dir, dataset):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, epoch, optimizer, scheduler, model_dir, dataset)

        elif self._is_not_improving(score):  # No improvement
            self.counter += 1
            print(f"Performance did not improve. Counter: {self.counter}/{self.patience}")
            self.save_checkpoint(model, epoch, optimizer, scheduler, model_dir, dataset)
            if self.counter >= self.patience:
                self.early_stop = True

        else:  # Improvement
            self.best_score = score
            self.save_checkpoint(model, epoch, optimizer, scheduler, model_dir, dataset)
            self.counter = 0  # Reset the counter

    def _is_not_improving(self, score):
        """Check if the score shows no improvement based on the mode."""
        if self.mode == 'min':  # Minimize (e.g., loss)
            return score > self.best_score + self.delta
        elif self.mode == 'max':  # Maximize (e.g., IoU)
            return score < self.best_score - self.delta
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
