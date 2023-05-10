import argparse
import math
import random
import shutil
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
import torchextractor as tx
from tensorboardX import SummaryWriter
from util import AverageMeter
from compressai.datasets import ImageFolder
import os
from torch.utils.data.distributed import DistributedSampler
from model_gdn import  Slimable_Modelgdn
from PyTorch_YOLOv3.pytorchyolo import models
from torch.nn.parallel import DistributedDataParallel
import imageio
from config import FLAGS

tb_logger = SummaryWriter('/...')
width_mult_list = [0.67, 1]
max_width = max(width_mult_list)
min_width = min(width_mult_list)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("-d", "--dataset", type=str, default="/data/coco", help="Training dataset")
    parser.add_argument("-e","--epochs",default=300,type=int,help="Number of epochs (default: %(default)s)",)
    parser.add_argument("-lr","--learning-rate",default=1e-4,type=float,help="Learning rate (default: %(default)s)",)
    parser.add_argument( "-n","--num-workers",type=int,default=4,help="Dataloaders threads (default: %(default)s)",)
    parser.add_argument( "--lambda",dest="lmbda",type=float,default=0.005,help="Bit-rate distortion parameter (default: %(default)s)",)
    parser.add_argument( "--lambda_fea",dest="lmbda_fea",type=float,default=0.0001,help="Bit-rate distortion parameter (default: %(default)s)",)
    parser.add_argument("--batch-size", type=int, default=48, help="Batch size (default: %(default)s)")
    parser.add_argument("--test-batch-size",type=int,default=48,help="Test batch size (default: %(default)s)",)
    parser.add_argument("--aux-learning-rate",default=1e-3,help="Auxiliary loss learning rate (default: %(default)s)",)
    parser.add_argument("--patch-size",type=int,nargs=2,default=(256, 256),help="Size of the patches to be cropped (default: %(default)s)",)
    parser.add_argument("--device", default="cuda", help="Use cuda")
    parser.add_argument("--seed", type=float, help="Set random seed for reproducibility")
    parser.add_argument("--clip_max_norm",default=1.0,type=float,help="gradient clipping max norm (default: %(default)s",)
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--local_rank", type=int, default=-1,help="node rank for distributed training")
    parser.add_argument("-m", "--model_yolo", type=str, default="PyTorch_YOLOv3/config/yolov3.cfg",
                        help="Path to model definition file (.cfg)")
    parser.add_argument("-w", "--weights", type=str, default="PyTorch_YOLOv3/weights/yolov3.weights",
                        help="Path to weights or checkpoint file (.weights or .pth)")
    args = parser.parse_args(argv)
    return args


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        
        out["mse_loss"] = self.mse(output['x_hat'], target)
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"]  + out["bpp_loss"] 

        return out

class RateFeatureLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda_fea=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda_fea

    def forward(self, output, target, imgs):
        N, _, H, W = imgs.size()
        out = {}
        num_pixels = N * H * W
        
        out["fea_loss"] = self.mse(output['fea'], target)
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        
        out["loss"] = self.lmbda * 255 ** 2 * out["fea_loss"]  + out["bpp_loss"] 

        return out


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar",):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "best_"+filename)


def train_one_epoch(
    model, model_yolo, criterion_recon, criterion_fea, train_dataloader, optimizer, aux_optimizer1, aux_optimizer2, epoch, args
):
    model.train()
    model_yolo.eval()
    device = next(model.parameters()).device

    for i, img in enumerate(train_dataloader):
        img = img.to(device)

        optimizer.zero_grad()
        aux_optimizer1.zero_grad()
        aux_optimizer2.zero_grad()

        with torch.no_grad():
            outputs_yolo = model_yolo(img)
            f_true = outputs_yolo[15]

        for width_mult in sorted(
                            width_mult_list, reverse=True):
            model.apply(lambda m: setattr(m, 'width_mult', width_mult))

            if width_mult == max_width:
                output_recon = model(img)
                loss_recon = criterion_recon(output_recon, img)
                loss_recon['loss'].backward()
                if args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.clip_max_norm)
                optimizer.step()
                aux_loss = model.module.aux_loss()
                aux_loss.backward()
                aux_optimizer1.step()
                
            else:
                output_fea = model(img)
                loss_fea = criterion_fea(output_fea, f_true, img)
                loss_fea['loss'].backward()
                optimizer.step()
                if args.clip_max_norm > 0: 
                    torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.clip_max_norm)
                aux_loss = model.module.aux_loss()
                aux_loss.backward() 
                aux_optimizer2.step()
                

        if i % 400 == 0 and args.local_rank == 0 :
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(img)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tMSE loss: {loss_recon["mse_loss"].item():.3f} |'
                f'\tBpp_all loss: {loss_recon["bpp_loss"].item():.3f} |'
                f'\tBpp_fea loss: {loss_fea["bpp_loss"].item():.3f} |'
                f'\tfea loss: {loss_fea["fea_loss"].item():.3f} '
            )



def test_epoch(epoch, test_dataloader, model, model_yolo, criterion_recon, criterion_fea, args):
    model.eval()
    model_yolo.eval()
    
    device = next(model.parameters()).device

    rd_loss = AverageMeter()
    bppall_loss = AverageMeter()
    bppfea_loss = AverageMeter()
    mse_loss = AverageMeter()
    fea_loss = AverageMeter()  
    aux_loss = AverageMeter()

    with torch.no_grad():
        for img in test_dataloader:
            img = img.to(device)
            with torch.no_grad():
                outputs_yolo = model_yolo(img)
                f_true = outputs_yolo[15]

            for width_mult in sorted(
                            width_mult_list, reverse=True):
                model.apply(
                    lambda m: setattr(m, 'width_mult', width_mult))

                if width_mult == max_width:
                    output_recon = model(img)
                    loss_recon = criterion_recon(output_recon, img)
                else:
                    output_fea = model(img)
                    loss_fea = criterion_fea(output_fea, f_true, img)

            loss = loss_recon['loss'] + loss_fea['loss']

            bppall_loss.update(loss_recon["bpp_loss"])
            bppfea_loss.update(loss_fea['fea_loss'])
            rd_loss.update(loss)
            mse_loss.update(loss_recon["mse_loss"])
            fea_loss.update(loss_fea["fea_loss"])
            aux_loss.update(model.module.aux_loss())

    if args.local_rank == 0 :
        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {rd_loss.avg:.3f} |"
            f"\tMSE loss: {mse_loss.avg:.3f} |"
            f"\tBppall loss: {bppall_loss.avg:.3f} |"
            f"\tBppfea loss: {bppfea_loss.avg:.2f} |"
            f'\tfea loss: {fea_loss.avg:.3f} |'
            f"\tAux loss: {aux_loss.avg:.2f}"
        )

    return rd_loss.avg, mse_loss.avg, bppall_loss.avg, bppfea_loss.avg, fea_loss.avg


def main(argv):
    args = parse_args(argv)

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )
    device = torch.device(f'cuda:{args.local_rank}')

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    
    train_transforms = transforms.Compose(
        [transforms.Resize(args.patch_size), transforms.ToTensor()])

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()])

    train_dataset = ImageFolder(args.dataset, split="train2014", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="val2014", transform=test_transforms)

    train_sampler = DistributedSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler
        )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False)

    model = Slimable_Modelgdn(192).to(device)
  
    parameters = set(p for n, p in model.named_parameters() if not n.endswith(".quantiles"))
    optimizer = optim.Adam(parameters, lr=1e-4)
    aux_parameters1 = set(p for n, p in model.named_parameters() if n.endswith("entropy_bottleneck.quantiles"))
    aux_optimizer1 = optim.Adam(aux_parameters1, lr=1e-3)
    aux_parameters2 = set(p for n, p in model.named_parameters() if n.endswith("entropy_bottleneck_fea.quantiles"))
    aux_optimizer2 = optim.Adam(aux_parameters2, lr=1e-3)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion_recon = RateDistortionLoss(lmbda=args.lmbda)
    criterion_fea = RateFeatureLoss(lmbda_fea= args.lmbda_fea)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    # 加载YOLOv3模型
    model_yolo = models.load_model(args.model_yolo, args.weights)
    model_yolo = model_yolo.to(args.device)
    
    model = DistributedDataParallel(model,device_ids=[args.local_rank],output_device=args.local_rank,\
         broadcast_buffers=False,find_unused_parameters=True)
    model_yolo = DistributedDataParallel(model_yolo,device_ids=[args.local_rank],output_device=args.local_rank, \
        find_unused_parameters=True)

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        if args.local_rank == 0: 
            print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            model,
            model_yolo,
            criterion_recon,
            criterion_fea,
            train_dataloader,
            optimizer,
            aux_optimizer1,
            aux_optimizer2,
            epoch,
            args
        )
        if args.local_rank == 0: 
            loss, mse_loss, bppall_loss, bppfea_loss, fea_loss = test_epoch(\
                epoch, test_dataloader, model, model_yolo, criterion_recon, criterion_fea,args)

            lr_scheduler.step(loss)

            tb_logger.add_scalar('rd_loss', loss, epoch)
            tb_logger.add_scalar('bpp_all', bppall_loss, epoch)
            tb_logger.add_scalar('bpp_fea', bppfea_loss, epoch)
            tb_logger.add_scalar('feature', fea_loss, epoch)
            tb_logger.add_scalar('mse_loss', mse_loss, epoch)

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.module.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                filename="./checkpoint.pth.tar"
            )

    tb_logger.close()


if __name__ == "__main__":
    main(sys.argv[1:])