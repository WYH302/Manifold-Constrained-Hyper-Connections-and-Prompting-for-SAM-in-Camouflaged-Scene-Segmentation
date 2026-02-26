


import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
from PIL import Image
from torchvision import transforms
from typing import Any, Optional, Tuple
from transformers import AutoTokenizer, BlipProcessor, BlipForConditionalGeneration, MambaModel
from utils_downstream.saliency_metric import cal_mae, cal_sm, cal_em, cal_wfm, cal_dice, cal_iou, cal_ber, cal_acc


from segment_anything.modeling.mcsam_integrated import create_integrated_model, MMSAM_Integrated


torch.manual_seed(2024)
np.random.seed(2024)
torch.cuda.empty_cache()


def find_latest_model(work_dir="/root/autodl-tmp/MMsam/work_dir"):
    
    model_patterns = [
        ,
        ,
        
    ]
    found_models = []
    for pattern in model_patterns:
        search_path = os.path.join(work_dir, "**", pattern)
        model_files = list(glob.glob(search_path, recursive=True))
        found_models.extend(model_files)
    found_models = list(set(found_models))
    if not found_models:
        return None
    found_models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return found_models[0]


def eval_psnr(loader, model, vlm_model, processor, mamba_model, tokenizer,
              eval_type=None, device=None, hyper_cond=None):
    
    model.eval()
    print(f"\n=== Start Evaluation (Integrated Model + RankDice-RMA) ===")

    pbar = tqdm(total=len(loader), leave=False, desc='Evaluation Progress')
    mae, sm, em, wfm, m_dice, m_iou, ber = (
        cal_mae(), cal_sm(), cal_em(), cal_wfm(),
        cal_dice(), cal_iou(), cal_ber()
    )

    if hyper_cond is None:
        hyper_cond = {'threshold': 0.5, 'boundary_weight': 1.0}

    with torch.no_grad():
        for step, (image, gt2D, img_1024_ori, img_path) in enumerate(loader):
            image, gt2D = image.to(device), gt2D.to(device)
            img_1024_ori = img_1024_ori.to(device)

            
            vlm_inputs = processor(img_1024_ori, return_tensors="pt").to(device)
            vlm_outputs = vlm_model.generate(**vlm_inputs)
            description = processor.decode(vlm_outputs[0], skip_special_tokens=True)

            mamba_inputs = tokenizer(description, padding=True, return_tensors="pt").to(device)
            mamba_outputs = mamba_model(**mamba_inputs)

            vision_outputs = vlm_model.vision_model(**vlm_inputs)
            image_features_raw = vision_outputs.last_hidden_state[:, 1:, :]  

            
            batch_size, seq_len, hidden_dim = image_features_raw.shape
            if seq_len == 576:  
                image_features = image_features_raw.reshape(batch_size, 24, 24, hidden_dim)
                image_features = image_features.permute(0, 3, 1, 2)               
                image_features = F.interpolate(image_features, size=(64, 64),
                                               mode='bilinear', align_corners=False)
            elif seq_len == 64 * 64:
                image_features = image_features_raw.reshape(batch_size, 64, 64, hidden_dim).permute(0, 3, 1, 2)
            else:
                image_features = image_features_raw.mean(dim=1, keepdim=True)    
                image_features = image_features.unsqueeze(-1)                    
                image_features = F.interpolate(image_features, size=(64, 64))   
                image_features = image_features.squeeze(1)                      

            text_features = mamba_outputs.last_hidden_state                     

            
            pred_mask, _, _ = model(
                image=image,
                text_embeddings=text_features,
                image_features=image_features,
                gt_mask=None,
                hyper_cond=hyper_cond,
                return_logits=False                     
            )

            
            pred_np = pred_mask.squeeze().cpu().numpy()
            gt_np = gt2D.squeeze().cpu().numpy()
            if pred_np.ndim == 0:
                pred_np = np.expand_dims(pred_np, 0)
            if gt_np.ndim == 0:
                gt_np = np.expand_dims(gt_np, 0)

            mae.update(pred_np, gt_np)
            sm.update(pred_np, gt_np)
            em.update(pred_np, gt_np)
            wfm.update(pred_np, gt_np)
            m_dice.update(pred_np, gt_np)
            m_iou.update(pred_np, gt_np)
            ber.update(pred_np, gt_np)

            pbar.set_description(f"Evaluation step {step + 1}/{len(loader)}")
            pbar.update(1)

    MAE = mae.show()
    sm_val = sm.show()
    em_val = em.show()
    wfm_val = wfm.show()
    dice_val = m_dice.show()
    iou_val = m_iou.show()
    ber_val = ber.show()
    pbar.close()
    return sm_val, em_val, wfm_val, MAE, dice_val, iou_val, ber_val


class PositionEmbeddingRandom(nn.Module):
    
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            ,
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)


class InferenceDataset(Dataset):
    
    def __init__(self, data_root):
        self.data_root = data_root
        img_dir = os.path.join(data_root, "Imgs")
        gt_dir = os.path.join(data_root, "GT")
        if not os.path.exists(img_dir):
            img_dir = data_root
            gt_dir = data_root

        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        self.img_files = []
        for ext in img_extensions:
            self.img_files.extend([f for f in os.listdir(img_dir) if f.lower().endswith(ext)])

        self.gt_files = []
        if os.path.exists(gt_dir):
            for ext in ['.png', '.jpg', '.jpeg']:
                self.gt_files.extend([f for f in os.listdir(gt_dir) if f.lower().endswith(ext)])

        self.img_files = sorted(self.img_files)
        self.gt_files = sorted(self.gt_files)
        self.img_dir = img_dir
        self.gt_dir = gt_dir

        print(f"Dataset: {data_root}")
        print(f"Image count: {len(self.img_files)}")
        print(f"GT count: {len(self.gt_files)}")

        self.img_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32)
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_filename = self.img_files[index]
        img_path = os.path.join(self.img_dir, img_filename)
        img_ori = Image.open(img_path).convert('RGB')
        img_tensor = self.img_transform(img_ori)

        if self.gt_files and index < len(self.gt_files):
            gt_filename = self.gt_files[index]
            gt_path = os.path.join(self.gt_dir, gt_filename)
            gt = Image.open(gt_path).convert('L')
            gt_tensor = self.mask_transform(gt)
        else:
            gt_tensor = torch.zeros((1, 1024, 1024))

        return img_tensor, gt_tensor, np.array(img_ori), img_path


def visualize_prediction(image, gt, pred, save_path=None):
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        pass
    elif len(image.shape) == 3 and image.shape[1] == 3:
        image = image.transpose(1, 2, 0)
    elif len(image.shape) == 4:
        image = image[0]
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        elif image.shape[2] == 3:
            pass
    if image.max() > 1.0:
        image = image / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    axes[1].imshow(gt, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title("Prediction")
    axes[2].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="MMSAM_Integrated Inference (4-in-1 integration)")
    
    parser.add_argument("-test_data_path", type=str,
                        default="/root/autodl-tmp/data/NC4K_Inference",
                        help="Test data path")
    parser.add_argument("-results_dir", type=str,
                        default="/root/autodl-tmp/MC-SAM/result/NC4K",
                        help="Results save directory")
    
    parser.add_argument("-model_type", type=str, default="vit_l",
                        choices=["vit_b", "vit_l", "vit_h"],
                        help="SAM model type")
    parser.add_argument("-sam_checkpoint", type=str,
                        default="/root/autodl-tmp/MMsam/sam/sam_vit_l_0b3195.pth",
                        help="SAM pre-trained weights path")
    parser.add_argument("-checkpoint", type=str,
                        default="/root/autodl-tmp/MC-SAM/work_dir/mmsam_integrated/20260211-0026/model_best.pth",
                        help="Checkpoint path for trained integrated model (searches automatically if not specified)")
    parser.add_argument("-blip_feature_dim", type=int, default=768,
                        help="BLIP vision feature dimension")
    parser.add_argument("-n_streams", type=int, default=4,
                        help="Number of streams for Manifold Constrained Adapter (must match training)")
    parser.add_argument("--use_rankdice", action="store_true", default=True,
                        help="Enable RankDice-RMA module")
    parser.add_argument("--use_hypercond", action="store_true", default=True,
                        help="Enable Hyperparameter Conditioning module")
    
    parser.add_argument("-device", type=str, default="cuda:0",
                        help="Device")
    parser.add_argument("-batch_size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("-num_workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("-save_results", action="store_true", default=True,
                        help="Save prediction results")
    parser.add_argument("-visualize", action="store_true", default=True,
                        help="Visualize and save results")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold in hyperparameter condition (effective when use_hypercond=True)")
    parser.add_argument("--boundary_weight", type=float, default=1.0,
                        help="Boundary weight in hyperparameter condition")

    args = parser.parse_args()

    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    if args.save_results:
        os.makedirs(args.results_dir, exist_ok=True)
        os.makedirs(os.path.join(args.results_dir, "predictions"), exist_ok=True)
        if args.visualize:
            os.makedirs(os.path.join(args.results_dir, "visualizations"), exist_ok=True)

    
    if not args.checkpoint:
        print("Checkpoint path not specified, attempting to search automatically...")
        args.checkpoint = find_latest_model()
        if not args.checkpoint:
            print("Error: No model files found")
            return
        print(f"Found model: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file does not exist: {args.checkpoint}")
        return

    
    print("Loading SAM pre-trained weights...")
    if not os.path.exists(args.sam_checkpoint):
        print(f"Warning: SAM weights not found: {args.sam_checkpoint}")
        return
    sam_model = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)

    
    print("Creating MMSAM_Integrated integrated model...")
    integrated_model = create_integrated_model(
        sam_checkpoint_path=args.sam_checkpoint,
        model_type=args.model_type,
        image_size=1024,
        use_rankdice=args.use_rankdice,
        use_hypercond=args.use_hypercond,
        n_streams=args.n_streams,
        device=device,
        blip_feature_dim=args.blip_feature_dim
    ).to(device)

    
    print(f"Loading training weights: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'N/A')
        print(f"Checkpoint epoch: {epoch}")
    else:
        state_dict = checkpoint

    
    missing_keys, unexpected_keys = integrated_model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys (usually prompt_encoder related, can be ignored): {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys[:5]}...")

    integrated_model.eval()
    print("Model loaded successfully!")

    
    print("Loading VLM (BLIP) and Mamba models...")
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    try:
        processor = BlipProcessor.from_pretrained("/root/autodl-tmp/MMsam/Blip")
        vlm_model = BlipForConditionalGeneration.from_pretrained("/root/autodl-tmp/MMsam/Blip").to(device)
        tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/MMsam/mamba")
        mamba_model = MambaModel.from_pretrained("/root/autodl-tmp/MMsam/mamba").to(device)
    except Exception as e:
        print(f"Local loading failed, attempting online loading: {e}")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        vlm_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
        tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
        mamba_model = MambaModel.from_pretrained("state-spaces/mamba-130m-hf").to(device)

    vlm_model.eval()
    mamba_model.eval()

    
    print(f"Loading test data: {args.test_data_path}")
    test_dataset = InferenceDataset(args.test_data_path)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"Test sample count: {len(test_dataset)}")

    
    if len(test_dataset.gt_files) > 0:
        print("\nStarting quantitative evaluation...")
        hyper_cond = {'threshold': args.threshold, 'boundary_weight': args.boundary_weight}
        metrics = eval_psnr(
            test_dataloader,
            integrated_model,
            vlm_model,
            processor,
            mamba_model,
            tokenizer,
            eval_type='cod',
            device=device,
            hyper_cond=hyper_cond
        )
        sm, em, wfm, mae, dice, iou, ber = metrics
        print("\n" + "=" * 50)
        print("Evaluation Results:")
        print("=" * 50)
        print(f"Sm (Structure Measure): {sm:.4f}")
        print(f"Em (Enhanced-alignment Measure): {em:.4f}")
        print(f"wFm (Weighted F-measure): {wfm:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Dice: {dice:.4f}")
        print(f"IoU: {iou:.4f}")
        print(f"BER: {ber:.4f}")
        print("=" * 50)
    else:
        print("Warning: GT files not found, skipping quantitative evaluation")

    
    if args.save_results:
        print("\nSaving prediction results...")
        integrated_model.eval()
        hyper_cond = {'threshold': args.threshold, 'boundary_weight': args.boundary_weight}

        with torch.no_grad():
            for batch_idx, (images, gts, images_ori, img_paths) in enumerate(tqdm(test_dataloader, desc="Saving Results")):
                images = images.to(device)

                if isinstance(images_ori, np.ndarray):
                    images_ori_tensor = torch.from_numpy(images_ori).permute(0, 3, 1, 2).to(device)
                else:
                    images_ori_tensor = images_ori.to(device)

                
                vlm_inputs = processor(images_ori_tensor, return_tensors="pt").to(device)
                vlm_outputs = vlm_model.generate(**vlm_inputs)
                descriptions = processor.batch_decode(vlm_outputs, skip_special_tokens=True)

                mamba_inputs = tokenizer(descriptions, padding=True, return_tensors="pt").to(device)
                mamba_outputs = mamba_model(**mamba_inputs)

                vision_outputs = vlm_model.vision_model(**vlm_inputs)
                image_features_raw = vision_outputs.last_hidden_state[:, 1:, :]

                
                batch_size, seq_len, hidden_dim = image_features_raw.shape
                if seq_len == 576:
                    image_features = image_features_raw.reshape(batch_size, 24, 24, hidden_dim)
                    image_features = image_features.permute(0, 3, 1, 2)
                    image_features = F.interpolate(image_features, size=(64, 64),
                                                   mode='bilinear', align_corners=False)
                else:
                    image_features = image_features_raw  

                text_features = mamba_outputs.last_hidden_state

                
                pred_masks, _, _ = integrated_model(
                    image=images,
                    text_embeddings=text_features,
                    image_features=image_features,
                    gt_mask=None,
                    hyper_cond=hyper_cond,
                    return_logits=False
                )

                
                for i in range(len(pred_masks)):
                    img_name = os.path.basename(img_paths[i])
                    base_name = os.path.splitext(img_name)[0]

                    pred_np = pred_masks[i].squeeze().cpu().numpy()
                    pred_img = Image.fromarray((pred_np * 255).astype(np.uint8))
                    pred_path = os.path.join(args.results_dir, "predictions", f"{base_name}_pred.png")
                    pred_img.save(pred_path)

                    if args.visualize:
                        gt_np = gts[i].squeeze().cpu().numpy()
                        if isinstance(images_ori, np.ndarray):
                            img_np = images_ori[i]
                            if img_np.shape[0] == 3:
                                img_np = img_np.transpose(1, 2, 0)
                        else:
                            img_np = images_ori_tensor[i].cpu().numpy()
                            if img_np.shape[0] == 3:
                                img_np = img_np.transpose(1, 2, 0)
                        if img_np.max() > 1:
                            img_np = img_np / 255.0
                        vis_path = os.path.join(args.results_dir, "visualizations", f"{base_name}_vis.png")
                        visualize_prediction(img_np, gt_np, pred_np, vis_path)

        print(f"Prediction results saved to: {args.results_dir}")

    print("\nInference completed!")


if __name__ == "__main__":
    main()