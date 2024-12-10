
import os
import torch

import numpy as np
import os.path as osp
import logging
from torch import nn, optim, utils
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm

from dataset.dataset import DiffMOTDataset
from models.autoencoder import D2MP
from models.condition_embedding import History_motion_embedding

import time
# from tracker.DiffMOTtracker import diffmottracker

from tracking_utils.log import logger
from tracking_utils.timer import Timer

# This line is for training
from utils import calculate_iou, calculate_ade, original_shape
from early_stopping import EarlyStopping

def write_results(filename, results, data_type='mot'):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

def custom_collate_fn(batch):
    for sample in batch:
        if 'image_path' in sample:
            del sample['image_path']
    return torch.utils.data.default_collate(batch)

class DiffMOT():
    def __init__(self, config):
        self.config = config
        torch.backends.cudnn.benchmark = True
        self._build()

    def generate(self, conds, sample = 1, bestof = True, flexibility = 0.0, ret_traj = False):
        cond_encodeds = self.model.encoder(conds)
        track_pred = self.model.diffusion.sample(cond_encodeds, sample, bestof, flexibility=flexibility, ret_traj=ret_traj)
        return track_pred.squeeze(dim=0)

    def step(self, data_loader, train=True):
        self.model.train() if train else self.model.eval()
        
        total_loss = 0
        total_iou = 0
        total_ade = 0
        num_batches = len(data_loader)
        
        for batch in tqdm(data_loader):
            for k in batch:
                batch[k] = batch[k].to(device=self.device, non_blocking=True)

            loss = self.model(batch)
            loss = loss.mean()
            
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()

            # MeanIoU and MeanADE
            with torch.no_grad():
                predictions = self.generate(conds = batch['condition'], sample = 1, bestof =  True, flexibility=0.0, ret_traj=False) # Batch_size, 4
            dets = batch['condition'][:, 4, :4] # Batch_size, 4
            predictions = predictions + dets # Batch_size, 4

            targets = batch['cur_bbox'] # Batch_size, 4
            width = batch['width'] # Batch_size
            height = batch['height'] # Batch_size

            original_preds = original_shape(predictions, width, height) # Batch_size, 4
            original_gts = original_shape(targets, width, height) # Batch_size, 4

            total_iou += calculate_iou(original_preds, original_gts)
            total_ade += calculate_ade(original_preds, original_gts)
        
        mean_loss = total_loss / num_batches
        mean_iou = total_iou / num_batches
        mean_ade = total_ade / num_batches
        
        return {
            'mean_loss': mean_loss,
            'mean_iou': mean_iou,
            'mean_ade': mean_ade
        }

    def train(self):
        for epoch in range(1, self.config.epochs + 1):
            print("Training")
            train_metrics = self.step(data_loader = self.train_dataloader, train=True)
            print("Validation")
            val_metrics = self.step(data_loader = self.val_dataloader, train=False)

            self.scheduler.step()

            print(f"Epoch {epoch}/{self.config.epochs}")
            print(f"Train - Loss: {train_metrics['mean_loss']:.6f}, IoU: {train_metrics['mean_iou']:.6f}, ADE: {train_metrics['mean_ade']:.6f}")
            print(f"Val   - Loss: {val_metrics['mean_loss']:.6f}, IoU: {val_metrics['mean_iou']:.6f}, ADE: {val_metrics['mean_ade']:.6f}")

            self.early_stopping(val_metrics['mean_loss'], self.model, epoch, self.optimizer, self.scheduler, self.model_dir, self.config.dataset)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

    # def eval(self):
    #     det_root = self.config.det_dir
    #     img_root = det_root.replace('/detections/', '/')

    #     seqs = [s for s in os.listdir(det_root)]
    #     seqs.sort()

    #     for seq in seqs:
    #         print(seq)
    #         det_path = osp.join(det_root, seq)
    #         img_path = osp.join(img_root, seq, 'img1')

    #         info_path = osp.join(self.config.info_dir, seq, 'seqinfo.ini')
    #         seq_info = open(info_path).read()
    #         seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    #         seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    #         tracker = diffmottracker(self.config)
    #         timer = Timer()
    #         results = []
    #         frame_id = 0

    #         frames = [s for s in os.listdir(det_path)]
    #         frames.sort()
    #         imgs = [s for s in os.listdir(img_path)]
    #         imgs.sort()

    #         for i, f in enumerate(frames):
    #             if frame_id % 10 == 0:
    #                 logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

    #             timer.tic()
    #             f_path = osp.join(det_path, f)
    #             dets = np.loadtxt(f_path, dtype=np.float32, delimiter=',').reshape(-1, 6)[:, 1:6]

    #             im_path = osp.join(img_path, imgs[i])
    #             # img = cv2.imread(im_path)
    #             tag = f"{seq}:{frame_id+1}"
    #             # track
    #             # online_targets = tracker.update(dets, self.model, frame_id, seq_width, seq_height, tag, img)
    #             online_targets = tracker.update(dets, self.model, frame_id, seq_width, seq_height, tag)
    #             online_tlwhs = []
    #             online_ids = []
    #             for t in online_targets:
    #                 tlwh = t.tlwh
    #                 tid = t.track_id
    #                 online_tlwhs.append(tlwh)
    #                 online_ids.append(tid)
    #             timer.toc()
    #             # save results
    #             results.append((frame_id + 1, online_tlwhs, online_ids))
    #             frame_id += 1

    #         tracker.dump_cache()
    #         result_root = self.config.save_dir
    #         mkdirs(result_root)
    #         result_filename = osp.join(result_root, '{}.txt'.format(seq))
    #         write_results(result_filename, results)


    def _build(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._build_dir()
        self._build_encoder()
        self._build_model()
        self._build_train_loader()
        self._build_optimizer()

        print("> Everything built. Have fun :)")

    def _build_dir(self):
        self.model_dir = osp.join("./experiments",self.config.eval_expname)
        self.log_writer = SummaryWriter(log_dir=self.model_dir)
        os.makedirs(self.model_dir,exist_ok=True)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        log_name = f"{self.config.dataset}_{log_name}"

        log_dir = osp.join(self.model_dir, log_name)
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        handler = logging.FileHandler(log_dir)
        handler.setLevel(logging.INFO)
        self.log.addHandler(handler)

        self.log.info("Config:")
        self.log.info(self.config)
        self.log.info("\n")
        self.log.info("Eval on:")
        self.log.info(self.config.dataset)
        self.log.info("\n")


        if self.config.eval_mode:
            epoch = self.config.eval_at
            checkpoint_dir = osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt")
            self.checkpoint = torch.load(checkpoint_dir, map_location = self.device)

        print("> Directory built!")

    def _build_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=0.98)
        self.early_stopping = EarlyStopping(patience=self.config.patience, delta=self.config.delta)
        print("> Optimizer built!")

    def _build_encoder(self):
        self.encoder = History_motion_embedding()


    def _build_model(self):
        """ Define Model """
        config = self.config
        model = D2MP(config, encoder=self.encoder)

        self.model = model
        self.model.to(self.device)

        # if not self.config.eval_mode:
        #     self.model = torch.nn.DataParallel(self.model, self.config.gpus).to('cuda')
        # else:
        #     self.model = self.model.cuda()
        #     self.model = self.model.eval()

        # if self.config.eval_mode:
        #     self.model.load_state_dict({k.replace('module.', ''): v for k, v in self.checkpoint['ddpm'].items()})

        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("> Model built!")
        print('Network Version: ', self.config.network)
        print(f'Network Params: {params}')

    def _build_train_loader(self):
        config = self.config
        data_path = config.data_dir

        train_path = f'{data_path}/train'
        print("Train Dataset: " + train_path)
        self.train_dataset = DiffMOTDataset(train_path, config)
        print("len: ", len(self.train_dataset))

        print("="*80)
        val_path = f'{data_path}/val'
        print("Validation Dataset: " + val_path)
        self.val_dataset = DiffMOTDataset(val_path, config)
        print("len: ", len(self.val_dataset))

        self.train_dataloader = utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            # num_workers=self.config.preprocess_workers,
            # pin_memory=True,
            collate_fn = custom_collate_fn
        )

        self.val_dataloader = utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            # num_workers=self.config.preprocess_workers,
            # pin_memory=True,
            collate_fn = custom_collate_fn
        )

    print("> Train Dataset built!")
