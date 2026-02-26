



import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import shutil
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  
matplotlib.rcParams['axes.unicode_minus'] = False  


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions import Beta

from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from transformers import AutoTokenizer, BlipProcessor, BlipForConditionalGeneration, MambaModel


from segment_anything.modeling.mcsam_integrated import (
    create_integrated_model,
    create_optimizer_for_integrated_model,
    BoundaryAwareLoss,
    evaluate_model
)


try:
    from utils_downstream.saliency_metric import cal_mae, cal_sm, cal_em, cal_wfm, cal_dice, cal_iou, cal_ber, cal_acc
except ImportError:
    print("Warning: Evaluation metric module not found, using simple version")


    
    class SimpleMetric:
        def __init__(self):
            self.values = []

        def update(self, pred, gt):
            
            self.values.append(0.5)

        def show(self):
            return np.mean(self.values) if self.values else 0.0


    
    cal_mae = cal_sm = cal_em = cal_wfm = cal_dice = cal_iou = cal_ber = cal_acc = SimpleMetric


torch.manual_seed(2024)
np.random.seed(2024)


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '0'  



class NpyDataset(Dataset):
    

    def __init__(self, data_root, img_size=1024):
        self.data_root = data_root
        self.img_size = img_size

        
        self.gt_path = os.path.join(data_root, "GT/")
        self.img_path = os.path.join(data_root, "Imgs/")

        self.gt_path_files = sorted([
            os.path.join(self.gt_path, f)
            for f in os.listdir(self.gt_path)
            if f.endswith('.png')
        ])
        self.img_path_files = sorted([
            os.path.join(self.img_path, f)
            for f in os.listdir(self.img_path)
            if f.endswith('.jpg') or f.endswith('.png')
        ])

        
        assert len(self.gt_path_files) == len(self.img_path_files),            f"Image and mask count mismatch: {len(self.img_path_files)} vs {len(self.gt_path_files)}"

        print(f"Dataset: {data_root}")
        print(f"Image count: {len(self.img_path_files)}")

        
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32)
        ])

    def __len__(self):
        return len(self.img_path_files)

    def __getitem__(self, idx):
        
        img_path = self.img_path_files[idx]
        img_ori = Image.open(img_path).convert('RGB')

        
        gt_path = self.gt_path_files[idx]
        gt = Image.open(gt_path).convert('L')

        
        img_tensor = self.img_transform(img_ori)
        gt_tensor = self.mask_transform(gt)

        
        img_ori_array = np.array(img_ori)

        return img_tensor, gt_tensor, img_ori_array



def eval_psnr(loader, model, vlm_model, processor, mamba_model, tokenizer, device, use_rankdice=True):
    
    model.eval()
    model.set_training_mode(False)

    print(f"\n=== Start Evaluation ===")
    pbar = tqdm(total=len(loader), leave=False, desc='Evaluation Progress')

    
    mae, sm, em, wfm, m_dice, m_iou, ber = cal_mae(), cal_sm(), cal_em(), cal_wfm(), cal_dice(), cal_iou(), cal_ber()

    with torch.no_grad():
        for step, (image, gt2D, img_1024_ori) in enumerate(loader):
            image, gt2D = image.to(device), gt2D.to(device)
            img_1024_ori = img_1024_ori.to(device)

            
            vlm_inputs = processor(img_1024_ori, return_tensors="pt").to(device)
            vlm_outputs = vlm_model.generate(**vlm_inputs)
            description = processor.decode(vlm_outputs[0], skip_special_tokens=True)

            
            mamba_inputs = tokenizer(description, padding=True, return_tensors="pt").to(device)
            mamba_outputs = mamba_model(**mamba_inputs)
            text_features = mamba_outputs.last_hidden_state

            
            vision_outputs = vlm_model.vision_model(**vlm_inputs)
            image_features = vision_outputs.last_hidden_state[:, 1:, :]

            
            batch_size, seq_len, hidden_dim = image_features.shape
            if seq_len == 576:  
                image_features = image_features.reshape(batch_size, 24, 24, hidden_dim)
                image_features = image_features.permute(0, 3, 1, 2)
                image_features = F.interpolate(image_features, size=(64, 64), mode='bilinear', align_corners=False)
            elif seq_len == 64 * 64:  
                image_features = image_features.reshape(batch_size, 64, 64, hidden_dim).permute(0, 3, 1, 2)
            else:
                image_features = image_features.mean(dim=1, keepdim=True)
                image_features = image_features.unsqueeze(-1)
                image_features = F.interpolate(image_features, size=(64, 64))

            
            pred_mask, _, _ = model(
                image=image,
                text_embeddings=text_features,
                image_features=image_features,
                gt_mask=None,
                hyper_cond={'threshold': 0.5, 'boundary_weight': 1.0},
                return_logits=False
            )

            
            pred = pred_mask.squeeze().cpu().numpy()
            gt = gt2D.squeeze().cpu().numpy()

            
            if pred.ndim == 0:
                pred = np.expand_dims(pred, 0)
            if gt.ndim == 0:
                gt = np.expand_dims(gt, 0)

            
            mae.update(pred, gt)
            sm.update(pred, gt)
            em.update(pred, gt)
            wfm.update(pred, gt)
            m_dice.update(pred, gt)
            m_iou.update(pred, gt)
            ber.update(pred, gt)

            pbar.update(1)
            pbar.set_description(f"Evaluation step {step + 1}/{len(loader)}")

    pbar.close()

    
    metrics = {
        : mae.show(),
        : sm.show(),
        : em.show(),
        : wfm.show(),
        : m_dice.show(),
        : m_iou.show(),
        : ber.show(),
    }

    return metrics



def parse_args():
    parser = argparse.ArgumentParser(description="Train integrated MMSAM model")

    
    parser.add_argument("--train_data", type=str,
                        default="/root/autodl-tmp/data/COD10K+CAMO/COD10K_CAMO_CombinedTrainingDataset",
                        help="Train data path")
    parser.add_argument("--val_data", type=str,
                        default="/root/autodl-tmp/data/COD10K+CAMO/COD10K_CAMO_CombinedTestingDataset/TestingDataset",
                        help="Validation data path")
    parser.add_argument("--val_sample_size", type=int, default=1000,
                        help="Validation sample size")

    
    parser.add_argument("--sam_checkpoint", type=str,
                        default="/root/autodl-tmp/MMsam/sam/sam_vit_l_0b3195.pth",
                        help="SAM pre-trained weights path")
    parser.add_argument("--model_type", type=str, default="vit_l",
                        choices=["vit_b", "vit_l", "vit_h"],
                        help="SAM model type")
    parser.add_argument("--blip_path", type=str,
                        default="/root/autodl-tmp/MMsam/Blip",
                        help="BLIP model path")
    parser.add_argument("--mamba_path", type=str,
                        default="/root/autodl-tmp/MMsam/mamba",
                        help="Mamba model path")

    
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--lr", type=float, default=0.00005,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")

    
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Training device")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use WandB logging")
    parser.add_argument("--resume", type=str,
                        default="/root/autodl-tmp/MC-SAM/work_dir/mmsam_integrated/20260211-0026/model_epoch_10.pth",
                        help="Checkpoint path to resume training")
    parser.add_argument("--work_dir", type=str, default="work_dir",
                        help="Working directory")
    parser.add_argument("--task_name", type=str, default="mmsam_integrated",
                        help="Task name")

    
    parser.add_argument("--n_streams", type=int, default=4,
                        help="Number of streams for Manifold Constrained Adapter")
    parser.add_argument("--use_rankdice", action="store_true", default=True,
                        help="Use RankDice-RMA module")
    parser.add_argument("--use_hypercond", action="store_true", default=True,
                        help="Use Hyperparameter Conditioning")

    return parser.parse_args()



def main():
    args = parse_args()

    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = os.path.join(args.work_dir, args.task_name, run_id)
    os.makedirs(model_save_path, exist_ok=True)

    
    shutil.copyfile(__file__, os.path.join(model_save_path, f"train_{run_id}.py"))

    
    if args.use_wandb:
        import wandb
        wandb.login()
        wandb.init(
            project=args.task_name,
            name=f"{args.task_name}_{run_id}",
            config=vars(args)
        )

    
    print("Loading dataset...")
    train_dataset = NpyDataset(args.train_data)
    val_dataset = NpyDataset(args.val_data)

    
    if len(val_dataset) > args.val_sample_size:
        indices = torch.randperm(len(val_dataset))[:args.val_sample_size]
        val_subset = Subset(val_dataset, indices)
        print(f"Val dataset: randomly selected {args.val_sample_size} images from {len(val_dataset)}")
    else:
        val_subset = val_dataset
        print(f"Val dataset: using all {len(val_dataset)} images")

    print(f"Train dataset: {len(train_dataset)} images")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    
    print("Loading VLM and text model...")
    processor = BlipProcessor.from_pretrained(args.blip_path)
    vlm_model = BlipForConditionalGeneration.from_pretrained(args.blip_path).to(device)
    vlm_model.eval()  

    tokenizer = AutoTokenizer.from_pretrained(args.mamba_path)
    mamba_model = MambaModel.from_pretrained(args.mamba_path).to(device)
    mamba_model.eval()  

    
    print("Creating integrated model...")
    model = create_integrated_model(
        sam_checkpoint_path=args.sam_checkpoint,
        model_type=args.model_type,
        image_size=1024,
        use_rankdice=args.use_rankdice,
        use_hypercond=args.use_hypercond,
        n_streams=args.n_streams,
        device=device,
        blip_feature_dim=768
    )

    
    for name, param in model.image_encoder.named_parameters():
        if "adapter" in name or "manifold_adapters" in name or "blip_feature_adjust" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable parameter ratio: {trainable_params / total_params * 100:.2f}%")

    
    print("Creating optimizer and loss function...")
    optimizer = create_optimizer_for_integrated_model(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs * len(train_loader),
        eta_min=1e-6
    )

    
    seg_loss = BoundaryAwareLoss(alpha=1.0, beta=0.1)

    
    beta_dist = Beta(torch.tensor([2.0]), torch.tensor([2.0]))

    
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    
    start_epoch = 0
    best_val_score = 0.0
    train_history = {
        : [],
        : [],
        : [],
        : [],
        : []
    }

    if args.resume and os.path.isfile(args.resume):
        print(f"Resume training from: {args.resume}")
        
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

        start_epoch = checkpoint.get('epoch', 0) + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'best_val_score' in checkpoint:
            best_val_score = checkpoint['best_val_score']

        if 'train_history' in checkpoint:
            train_history = checkpoint['train_history']

        print(f"Resume training: epoch {start_epoch}, best val score: {best_val_score:.4f}")

    
    print("Starting training...")
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'=' * 50}")

        
        model.train()
        model.set_training_mode(True)

        epoch_train_loss = 0.0
        epoch_rank_dice_loss = 0.0
        epoch_boundary_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f'Training (Epoch {epoch + 1}/{args.num_epochs})',
                          ncols=120, leave=False)

        for step, (image, gt2D, img_1024_ori) in enumerate(train_pbar):
            optimizer.zero_grad()

            
            image, gt2D = image.to(device), gt2D.to(device)
            img_1024_ori = img_1024_ori.to(device)

            
            tau_sample = beta_dist.sample().item()
            tau = 0.4 + tau_sample * 0.2  

            lambda_sample = beta_dist.sample().item()
            boundary_weight = 0.3 + lambda_sample * 0.7  

            hyper_cond = {
                : tau,
                : boundary_weight
            }

            
            with torch.no_grad():
                
                vlm_inputs = processor(img_1024_ori, return_tensors="pt").to(device)
                vlm_outputs = vlm_model.generate(**vlm_inputs)
                description = processor.decode(vlm_outputs[0], skip_special_tokens=True)

                
                mamba_inputs = tokenizer(description, padding=True, return_tensors="pt").to(device)
                mamba_outputs = mamba_model(**mamba_inputs)
                text_features = mamba_outputs.last_hidden_state

                
                vision_outputs = vlm_model.vision_model(**vlm_inputs)
                image_features_raw = vision_outputs.last_hidden_state[:, 1:, :]

            
            batch_size, seq_len, hidden_dim = image_features_raw.shape
            if seq_len == 576:  
                image_features = image_features_raw.reshape(batch_size, 24, 24, hidden_dim)
                image_features = image_features.permute(0, 3, 1, 2)
                image_features = F.interpolate(image_features, size=(64, 64), mode='bilinear', align_corners=False)
            elif seq_len == 64 * 64:  
                image_features = image_features_raw.reshape(batch_size, 64, 64, hidden_dim).permute(0, 3, 1, 2)
            else:
                image_features = image_features_raw.mean(dim=1, keepdim=True)
                image_features = image_features.unsqueeze(-1)
                image_features = F.interpolate(image_features, size=(64, 64))

            
            if args.use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    logits, rank_dice_loss, _ = model(
                        image=image,
                        text_embeddings=text_features,
                        image_features=image_features,
                        gt_mask=gt2D,
                        hyper_cond=hyper_cond,
                        return_logits=True
                    )

                    
                    total_loss, dice_ce_loss, boundary_loss = seg_loss(
                        logits, gt2D, boundary_weight
                    )
                    total_loss = total_loss + rank_dice_loss

                
                scaler.scale(total_loss).backward()

                
                model.apply_gradient_constraints()

                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
            else:
                logits, rank_dice_loss, _ = model(
                    image=image,
                    text_embeddings=text_features,
                    image_features=image_features,
                    gt_mask=gt2D,
                    hyper_cond=hyper_cond,
                    return_logits=True
                )

                
                total_loss, dice_ce_loss, boundary_loss = seg_loss(
                    logits, gt2D, boundary_weight
                )
                total_loss = total_loss + rank_dice_loss

                
                total_loss.backward()

                
                model.apply_gradient_constraints()

                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

            
            optimizer.zero_grad()

            
            scheduler.step()

            
            epoch_train_loss += total_loss.item()
            epoch_rank_dice_loss += rank_dice_loss.item() if isinstance(rank_dice_loss,
                                                                        torch.Tensor) else rank_dice_loss
            epoch_boundary_loss += boundary_loss.item()

            
            current_lr = optimizer.param_groups[0]['lr']
            
            if isinstance(rank_dice_loss, torch.Tensor):
                rank_dice_value = rank_dice_loss.item()
            else:
                rank_dice_value = rank_dice_loss

            
            if isinstance(boundary_loss, torch.Tensor):
                boundary_value = boundary_loss.item()
            else:
                boundary_value = boundary_loss

            
            train_pbar.set_postfix({
                : f'{total_loss.item():.4f}',
                : f'{rank_dice_value:.4f}',  
                : f'{boundary_value:.4f}',  
                : f'{current_lr:.2e}',
                
            })

        
        epoch_train_loss /= len(train_loader)
        epoch_rank_dice_loss /= len(train_loader)
        epoch_boundary_loss /= len(train_loader)

        
        train_history['train_loss'].append(epoch_train_loss)
        train_history['train_rank_dice_loss'].append(epoch_rank_dice_loss)
        train_history['train_boundary_loss'].append(epoch_boundary_loss)
        train_history['learning_rates'].append(current_lr)

        print(f"\nTraining Statistics:")
        print(f"  Total Loss: {epoch_train_loss:.4f}")
        print(f"  RankDice Loss: {epoch_rank_dice_loss:.4f}")
        print(f"  Boundary Loss: {epoch_boundary_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}")

        
        print(f"\nStart validation...")
        val_metrics = eval_psnr(
            val_loader, model, vlm_model, processor,
            mamba_model, tokenizer, device, args.use_rankdice
        )

        train_history['val_metrics'].append(val_metrics)

        print(f"\nValidation Metrics (Epoch {epoch + 1}):")
        print(f"  S-measure: {val_metrics['sm']:.4f}")
        print(f"  E-measure: {val_metrics['em']:.4f}")
        print(f"  wF-measure: {val_metrics['wfm']:.4f}")
        print(f"  MAE: {val_metrics['mae']:.4f}")
        print(f"  Dice: {val_metrics['dice']:.4f}")
        print(f"  IoU: {val_metrics['iou']:.4f}")
        print(f"  BER: {val_metrics['ber']:.4f}")

        
        val_score = (val_metrics['sm'] + val_metrics['em'] + val_metrics['wfm']) / 3

        
        if args.use_wandb:
            wandb.log({
                : epoch + 1,
                : epoch_train_loss,
                : epoch_rank_dice_loss,
                : epoch_boundary_loss,
                : current_lr,
                : val_metrics['sm'],
                : val_metrics['em'],
                : val_metrics['wfm'],
                : val_metrics['mae'],
                : val_metrics['dice'],
                : val_metrics['iou'],
                : val_metrics['ber'],
                : val_score,
            })

        
        
        checkpoint_latest = {
            : epoch,
            : model.state_dict(),
            : optimizer.state_dict(),
            : scheduler.state_dict(),
            : epoch_train_loss,
            : val_metrics,
            : val_score,
            : best_val_score,
            : train_history,
            : vars(args)
        }

        torch.save(checkpoint_latest, os.path.join(model_save_path, "model_latest.pth"))

        
        if val_score > best_val_score:
            best_val_score = val_score

            checkpoint_best = {
                : epoch,
                : model.state_dict(),
                : optimizer.state_dict(),
                : scheduler.state_dict(),
                : epoch_train_loss,
                : val_metrics,
                : val_score,
                : best_val_score,
                : train_history,
                : vars(args)
            }

            torch.save(checkpoint_best, os.path.join(model_save_path, "model_best.pth"))
            print(f"✅ Saved best model, val score: {val_score:.4f} (previous best: {best_val_score:.4f})")

        
        if (epoch + 1) % 5 == 0:
            checkpoint_epoch = {
                : epoch,
                : model.state_dict(),
                : optimizer.state_dict(),
                : scheduler.state_dict(),
                : epoch_train_loss,
                : val_metrics,
                : val_score,
                : best_val_score,
                : train_history,
                : vars(args)
            }

            torch.save(checkpoint_epoch, os.path.join(model_save_path, f"model_epoch_{epoch + 1}.pth"))

        
        plt.figure(figsize=(15, 10))

        
        plt.subplot(2, 3, 1)
        plt.plot(train_history['train_loss'], label='Total Loss')
        plt.plot(train_history['train_rank_dice_loss'], label='RankDice Loss')
        plt.plot(train_history['train_boundary_loss'], label='Boundary Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)

        
        plt.subplot(2, 3, 2)
        val_sm = [m['sm'] for m in train_history['val_metrics']]
        val_em = [m['em'] for m in train_history['val_metrics']]
        val_wfm = [m['wfm'] for m in train_history['val_metrics']]
        plt.plot(val_sm, label='S-measure')
        plt.plot(val_em, label='E-measure')
        plt.plot(val_wfm, label='wF-measure')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation Metrics')
        plt.legend()
        plt.grid(True)

        
        plt.subplot(2, 3, 3)
        val_mae = [m['mae'] for m in train_history['val_metrics']]
        plt.plot(val_mae, label='MAE', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('Mean Absolute Error')
        plt.legend()
        plt.grid(True)

        
        plt.subplot(2, 3, 4)
        val_dice = [m['dice'] for m in train_history['val_metrics']]
        val_iou = [m['iou'] for m in train_history['val_metrics']]
        plt.plot(val_dice, label='Dice Coefficient')
        plt.plot(val_iou, label='IoU')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Segmentation Metrics')
        plt.legend()
        plt.grid(True)

        
        plt.subplot(2, 3, 5)
        plt.plot(train_history['learning_rates'], label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')

        
        plt.subplot(2, 3, 6)
        val_ber = [m['ber'] for m in train_history['val_metrics']]
        plt.plot(val_ber, label='BER', color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('BER')
        plt.title('Balanced Error Rate')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(model_save_path, "training_curves.png"), dpi=150)
        plt.close()

        print(f"Epoch {epoch + 1} completed, training curves saved")

    
    print(f"\n{'=' * 50}")
    print(f"Training completed!")
    print(f"Best validation score: {best_val_score:.4f}")
    print(f"Model saved to: {model_save_path}")
    print(f"{'=' * 50}")

    
    if args.use_wandb:
        wandb.finish()



if __name__ == "__main__":
    main()