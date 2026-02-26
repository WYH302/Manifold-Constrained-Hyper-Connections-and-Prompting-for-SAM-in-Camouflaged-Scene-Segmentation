
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import math
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
from datetime import datetime
from PIL import Image
from torchvision import transforms
from typing import Any, Optional, Tuple, Type
from transformers import AutoTokenizer, AutoProcessor, MambaModel, BlipProcessor, BlipForConditionalGeneration


from segment_anything.modeling.image_encoder import ImageEncoderViT
from segment_anything.modeling.lora_module import LoRALinear, SharedLoRAAdapter


torch.manual_seed(2023)
torch.cuda.empty_cache()


class PositionEmbeddingRandom(nn.Module):
    

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
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
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)


class VLSAM(nn.Module):
    

    def __init__(self, image_encoder, mask_decoder, adapter_dim=256, image_feature_dim=768):
        super().__init__()
        
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.pe_layer = PositionEmbeddingRandom(256 // 2)

        
        self.text_adapter = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU()
        )

        
        self.image_feature_adapter = nn.Sequential(
            nn.Linear(image_feature_dim, adapter_dim),
            nn.GELU()
        )

        self.pseudo_mask_embed = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.GELU()
        )

    def forward(self, image, text_embeddings, image_features):
        
        if len(image_features.shape) == 3:
            batch_size, seq_len, hidden_dim = image_features.shape
            if seq_len == 576:
                h = w = 24
                image_features = image_features.reshape(batch_size, h, w, hidden_dim).permute(0, 3, 1, 2)
                image_features = F.interpolate(image_features, size=(64, 64), mode='bilinear', align_corners=False)
            elif seq_len == 64 * 64:
                h = w = 64
                image_features = image_features.reshape(batch_size, h, w, hidden_dim).permute(0, 3, 1, 2)
            else:
                image_features = image_features.mean(dim=1, keepdim=True).unsqueeze(-1)
                image_features = F.interpolate(image_features, size=(64, 64))

        
        image_embedding = self.image_encoder(image, image_features)

        
        text_features = self.text_adapter(text_embeddings)
        text_global = text_features.mean(dim=1, keepdim=True)

        if len(image_features.shape) == 4:
            image_global = image_features.mean(dim=[2, 3], keepdim=False)
        else:
            image_global = image_features.mean(dim=[1, 2], keepdim=False)

        image_global = self.image_feature_adapter(image_global)
        image_global = image_global.unsqueeze(1)

        sparse_embeddings = torch.cat([text_global, image_global], dim=1)
        dense_embeddings = self.pseudo_mask_embed(image_embedding)
        image_pe = self.pe_layer((64, 64)).unsqueeze(0)

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        return ori_res_masks


def find_latest_model(work_dir="work_dir"):
    
    model_patterns = ["vlsam_model_best.pth", "vlsam_model_latest.pth", "*.pth"]

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


class VLSAMPredictor:
    

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        
        self.model = self.load_model()

        
        self.vlm_processor, self.vlm_model, self.tokenizer, self.mamba_model = self.load_multimodal_models()

        
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print(f"VLSAM Predictor initialized, device: {self.device}")

    def load_model(self):
        
        print("Loading SAM base model...")

        
        checkpoint_path = self.args.checkpoint or find_latest_model()
        if not checkpoint_path:
            raise FileNotFoundError("Model checkpoint file not found")

        print(f"Analyzing checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        
        inferred_type = "vit_l"  
        for key in state_dict.keys():
            if "pos_embed" in key and state_dict[key].dim() == 4:
                embed_dim = state_dict[key].shape[-1]
                if embed_dim == 1024:
                    inferred_type = "vit_l"
                    print(f"Inferred vit_l model (embed_dim=1024)")
                elif embed_dim == 1280:
                    inferred_type = "vit_h"
                    print(f"Inferred vit_h model (embed_dim=1280)")
                elif embed_dim == 768:
                    inferred_type = "vit_b"
                    print(f"Inferred vit_b model (embed_dim=768)")
                break

        
        print(f"Using model type: {inferred_type}")

        
        if inferred_type == "vit_h":
            sam_checkpoint = "/root/autodl-tmp/MMSam_Lora/sam/sam_vit_h_4b8939.pth"
        elif inferred_type == "vit_l":
            sam_checkpoint = "/root/autodl-tmp/MMSam_Lora/sam/sam_vit_l_0b3195.pth"
        else:  
            sam_checkpoint = "/root/autodl-tmp/MMSam_Lora/sam/sam_vit_b_01ec64.pth"

        print(f"Using SAM checkpoint: {sam_checkpoint}")
        sam_model = sam_model_registry[inferred_type](checkpoint=sam_checkpoint)

        
        print("Creating VLSAM model...")
        model = VLSAM(
            image_encoder=sam_model.image_encoder,
            mask_decoder=sam_model.mask_decoder,
            adapter_dim=256,
            image_feature_dim=768
        ).to(self.device)

        
        print("Loading VLSAM weights...")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model.eval()

        
        adapter_count = 0
        for name, param in model.named_parameters():
            if "adapter" in name or "lora" in name:
                adapter_count += 1
                print(f"  Adapter parameter: {name} - {param.shape}")

        print(f"Loading complete! Total {adapter_count} adapter parameters loaded")

        return model

    def load_multimodal_models(self):
        
        print("Loading multi-modal models...")

        
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

        try:
            
            vlm_processor = BlipProcessor.from_pretrained("/root/autodl-tmp/MMsam/Blip")
            vlm_model = BlipForConditionalGeneration.from_pretrained("/root/autodl-tmp/MMsam/Blip").to(self.device)
            tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/MMsam/mamba")
            mamba_model = MambaModel.from_pretrained("/root/autodl-tmp/MMsam/mamba").to(self.device)
        except Exception as e:
            print(f"Local loading failed: {e}")
            print("Attempting online loading...")
            try:
                vlm_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
                vlm_model = BlipForConditionalGeneration.from_pretrained(
                    
                ).to(self.device)
                tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
                mamba_model = MambaModel.from_pretrained("state-spaces/mamba-130m-hf").to(self.device)
            except Exception as e2:
                print(f"Online loading failed: {e2}")
                raise

        vlm_model.eval()
        mamba_model.eval()

        return vlm_processor, vlm_model, tokenizer, mamba_model

    def process_image_features(self, image_features):
        
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

        return image_features

    def predict_single(self, image_path):
        
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        image_array = np.array(image)

        with torch.no_grad():
            
            vlm_inputs = self.vlm_processor(image, return_tensors="pt").to(self.device)
            vlm_outputs = self.vlm_model.generate(**vlm_inputs)
            description = self.vlm_processor.decode(vlm_outputs[0], skip_special_tokens=True)

            
            mamba_inputs = self.tokenizer(description, padding=True, return_tensors="pt").to(self.device)
            mamba_outputs = self.mamba_model(**mamba_inputs)

            
            vision_outputs = self.vlm_model.vision_model(**vlm_inputs)
            image_features = vision_outputs.last_hidden_state[:, 1:, :]
            image_features = self.process_image_features(image_features)

            
            text_features = mamba_outputs.last_hidden_state

            
            pred = torch.sigmoid(self.model(image_tensor, text_features, image_features))

            
            pred_np = pred.squeeze().cpu().numpy()
            pred_mask = (pred_np > 0.5).astype(np.uint8) * 255

        return pred_mask, description

    def predict_batch(self, image_paths):
        
        results = []

        for image_path in tqdm(image_paths, desc="Batch Predicting"):
            try:
                pred_mask, description = self.predict_single(image_path)
                results.append({
                    : image_path,
                    : pred_mask,
                    : description
                })
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                results.append({
                    : image_path,
                    : str(e)
                })

        return results

    def visualize(self, image_path, pred_mask, save_path=None):
        
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        
        axes[0].imshow(image_array)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        
        axes[1].imshow(pred_mask, cmap='gray')
        axes[1].set_title("Predicted Mask")
        axes[1].axis('off')

        
        axes[2].imshow(image_array)
        axes[2].imshow(pred_mask, cmap='jet', alpha=0.5)
        axes[2].set_title("Overlay Display")
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class MetricsCalculator:
    

    @staticmethod
    def compute_mae(pred, gt):
        
        return torch.abs(pred - gt).mean().item()

    @staticmethod
    def compute_max_fmeasure(pred, gt, beta2=0.3):
        
        pred = pred.float()
        gt = gt.float()

        if pred.max() == pred.min():
            pred = (pred > 0).float()

        prec = []
        recall = []

        thresholds = torch.linspace(0, 1, 256)
        for th in thresholds:
            binary_pred = (pred > th).float()
            tp = (binary_pred * gt).sum()
            fp = binary_pred.sum() - tp
            fn = gt.sum() - tp

            p = tp / (tp + fp + 1e-7)
            r = tp / (tp + fn + 1e-7)

            prec.append(p.item())
            recall.append(r.item())

        prec = torch.tensor(prec)
        recall = torch.tensor(recall)

        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall + 1e-7)

        return f_score.max().item()

    @staticmethod
    def compute_smeasure(pred, gt):
        
        alpha = 0.5
        y = gt.mean()
        if y == 0:
            x = pred.mean()
            Q = 1.0 - x
        elif y == 1:
            x = pred.mean()
            Q = x
        else:
            gt[gt >= 0.5] = 1
            gt[gt < 0.5] = 0
            Q = alpha * MetricsCalculator._s_object(pred, gt) + (1 - alpha) * MetricsCalculator._s_region(pred, gt)
            if Q < 0:
                Q = 0

        return Q.item()

    @staticmethod
    def _s_object(pred, gt):
        fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
        bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
        o_fg = MetricsCalculator._object(fg, gt)
        o_bg = MetricsCalculator._object(bg, 1 - gt)
        u = gt.mean()
        Q = u * o_fg + (1 - u) * o_bg
        return Q

    @staticmethod
    def _object(pred, gt):
        temp = pred[gt == 1]
        if temp.size(0) == 0:
            return 0
        x = temp.mean()
        sigma_x = temp.std()
        return 2.0 * x / (x * x + 1.0 + sigma_x + 1e-7)

    @staticmethod
    def _s_region(pred, gt):
        X, Y = MetricsCalculator._centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = MetricsCalculator._divideGT(gt, X, Y)
        p1, p2, p3, p4 = MetricsCalculator._dividePrediction(pred, X, Y)
        Q1 = MetricsCalculator._ssim(p1, gt1)
        Q2 = MetricsCalculator._ssim(p2, gt2)
        Q3 = MetricsCalculator._ssim(p3, gt3)
        Q4 = MetricsCalculator._ssim(p4, gt4)
        Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
        return Q

    @staticmethod
    def _centroid(gt):
        rows, cols = gt.size()
        total = gt.sum()
        if total == 0:
            X = torch.tensor(cols // 2, dtype=torch.float32)
            Y = torch.tensor(rows // 2, dtype=torch.float32)
        else:
            i = torch.arange(1, cols + 1, dtype=torch.float32).view(1, -1).repeat(rows, 1)
            j = torch.arange(1, rows + 1, dtype=torch.float32).view(-1, 1).repeat(1, cols)
            X = torch.round((gt * i).sum() / total)
            Y = torch.round((gt * j).sum() / total)
        return int(X.item()), int(Y.item())

    @staticmethod
    def _divideGT(gt, X, Y):
        h, w = gt.size()
        area = h * w
        LT = gt[:Y, :X]
        RT = gt[:Y, X:w]
        LB = gt[Y:h, :X]
        RB = gt[Y:h, X:w]

        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = (w - X) * (h - Y) / area

        return LT, RT, LB, RB, w1, w2, w3, w4

    @staticmethod
    def _dividePrediction(pred, X, Y):
        h, w = pred.size()
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]
        return LT, RT, LB, RB

    @staticmethod
    def _ssim(pred, gt):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = pred.mean()
        mu_y = gt.mean()
        sigma_x = pred.std()
        sigma_y = gt.std()
        sigma_xy = ((pred - mu_x) * (gt - mu_y)).mean()

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2)
        SSIM = SSIM_n / SSIM_d

        return SSIM.item()

    @staticmethod
    def compute_evaluation_all(pred, gt):
        
        metrics = {}

        
        metrics['MAE'] = MetricsCalculator.compute_mae(pred, gt)

        
        metrics['maxF'] = MetricsCalculator.compute_max_fmeasure(pred, gt)

        
        metrics['Smeasure'] = MetricsCalculator.compute_smeasure(pred, gt)

        
        metrics['Emeasure'] = MetricsCalculator.compute_emeasure(pred, gt)

        return metrics

    @staticmethod
    def compute_emeasure(pred, gt):
        
        pred = pred.numpy()
        gt = gt.numpy().astype(np.uint8)

        if np.max(gt) == 0:
            enhanced_mask = pred
        else:
            enhanced_mask = (pred - np.mean(pred)) * (np.std(gt) / (np.std(pred) + 1e-7)) + np.mean(gt)

        enhanced_mask = np.clip(enhanced_mask, 0, 1)

        th = 2 * np.mean(enhanced_mask)
        if th > 1:
            th = 1

        em = enhanced_mask >= th

        FM = np.sum(em & gt) / (np.sum(em | gt) + 1e-7)

        return float(FM)


class InferenceDataset(Dataset):
    

    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root

        
        self.img_files = []
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

        for ext in img_extensions:
            self.img_files.extend(glob.glob(os.path.join(data_root, 'Imgs', f'*{ext}')))
            self.img_files.extend(glob.glob(os.path.join(data_root, f'*{ext}')))

        
        self.img_files = sorted(list(set(self.img_files)))

        
        self.gt_files = []
        for img_path in self.img_files:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            
            gt_paths = [
                os.path.join(data_root, 'GT', f'{img_name}.png'),
                os.path.join(data_root, 'GT', f'{img_name}.jpg'),
                os.path.join(data_root, 'GT', f'{img_name}_mask.png'),
                os.path.join(data_root, 'GT', f'{img_name}.bmp'),
                img_path.replace('Imgs', 'GT').replace('.jpg', '.png'),
                img_path.replace('Imgs', 'GT')
            ]

            gt_found = False
            for gt_path in gt_paths:
                if os.path.exists(gt_path):
                    self.gt_files.append(gt_path)
                    gt_found = True
                    break

            if not gt_found:
                self.gt_files.append(None)

        
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.transform_ori = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])

        
        self.gt_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])

        print(f"Loading dataset: {data_root}")
        print(f"Found {len(self.img_files)} images")
        print(f"Found {len([gt for gt in self.gt_files if gt is not None])} GT files")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert('RGB')

        
        img_tensor = self.transform(img)
        img_ori_tensor = self.transform_ori(img)

        
        gt_path = self.gt_files[idx]
        if gt_path is not None and os.path.exists(gt_path):
            try:
                gt = Image.open(gt_path).convert('L')
                gt_tensor = self.gt_transform(gt)
                
                gt_tensor = (gt_tensor > 0.5).float()
            except:
                gt_tensor = torch.zeros((1, 1024, 1024))
        else:
            gt_tensor = torch.zeros((1, 1024, 1024))

        return img_tensor, gt_tensor, img_ori_tensor, img_path

def main():
    
    
    parser = argparse.ArgumentParser(description="VLSAM Model Inference")
    parser.add_argument("--test_data_path", type=str, default="/root/autodl-tmp/data/CHAMELEON_Inference",
                        help="Test data path")
    parser.add_argument("--image_dir", type=str, help="Input image directory (for batch prediction)")
    parser.add_argument("--single_image_path", type=str, help="Single image path")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/MMSam_Lora/result/CHAMELEON_SharedLoRA",
                        help="Results save directory")
    parser.add_argument("--model_type", type=str, default="vit_l", help="SAM model type")
    parser.add_argument("--checkpoint", type=str,
                        default="/root/autodl-tmp/MMSam_Lora/.\\work_dir/mmsam/vlsam_model_best.pth",
                        help="VLSAM model checkpoint path")
    parser.add_argument("--sam_checkpoint", type=str,
                        default="/root/autodl-tmp/MMSam_Lora/sam/sam_vit_l_0b3195.pth",
                        help="SAM pre-trained weights path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device used")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader worker threads")
    parser.add_argument("--save_results", action="store_true", default=True,
                        help="Whether to save prediction results")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Whether to visualize results")
    parser.add_argument("--image_feature_dim", type=int, default=768,
                        help="Image feature dimension")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--adapter_channels", type=int, default=256, help="Number of adapter channels")

    args = parser.parse_args()

    
    args.checkpoint = args.checkpoint.replace('\\.', '.')

    
    print("=" * 60)
    print("VLSAM Inference Configuration")
    print("=" * 60)
    print(f"Test data path: {args.test_data_path}")
    if args.single_image_path:
        print(f"Single image path: {args.single_image_path}")
    if args.image_dir:
        print(f"Image directory: {args.image_dir}")
    print(f"Model type: {args.model_type}")
    print(f"VLSAM checkpoint: {args.checkpoint}")
    print(f"SAM checkpoint: {args.sam_checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print("=" * 60)

    
    try:
        predictor = VLSAMPredictor(args)
        print("Predictor initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
        import traceback
        traceback.print_exc()
        return

    
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "predictions"), exist_ok=True)
        if args.visualize:
            os.makedirs(os.path.join(args.output_dir, "visualizations"), exist_ok=True)

    
    if args.single_image_path:
        
        if not os.path.exists(args.single_image_path):
            print(f"Error: Image file does not exist: {args.single_image_path}")
            return

        print(f"Processing single image: {args.single_image_path}")
        try:
            pred_mask, description = predictor.predict_single(args.single_image_path)

            
            base_name = os.path.splitext(os.path.basename(args.single_image_path))[0]

            
            mask_path = os.path.join(args.output_dir, "predictions", f"{base_name}_mask.png")
            Image.fromarray(pred_mask).save(mask_path)
            print(f"Saving mask to: {mask_path}")

            
            desc_path = os.path.join(args.output_dir, "predictions", f"{base_name}_description.txt")
            with open(desc_path, 'w', encoding='utf-8') as f:
                f.write(description)
            print(f"Saving description to: {desc_path}")
            print(f"Image description: {description}")

            
            if args.visualize:
                vis_path = os.path.join(args.output_dir, "visualizations", f"{base_name}_vis.png")
                predictor.visualize(args.single_image_path, pred_mask, save_path=vis_path)
                print(f"Saving visualization to: {vis_path}")

        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()

    elif args.image_dir:
        
        if not os.path.exists(args.image_dir):
            print(f"Error: Image directory does not exist: {args.image_dir}")
            return

        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(args.image_dir, f"*{ext}")))
            image_paths.extend(glob.glob(os.path.join(args.image_dir, f"*{ext.upper()}")))

        image_paths = sorted(list(set(image_paths)))

        if not image_paths:
            print(f"No image files found in directory: {args.image_dir}")
            return

        print(f"Found {len(image_paths)} images, starting batch processing...")

        results = predictor.predict_batch(image_paths)

        
        successful = 0
        for result in results:
            if 'error' in result:
                print(f"Failed: {result['path']} - {result['error']}")
                continue

            successful += 1
            image_path = result['path']
            pred_mask = result['mask']
            description = result['description']

            
            base_name = os.path.splitext(os.path.basename(image_path))[0]

            
            mask_path = os.path.join(args.output_dir, "predictions", f"{base_name}_mask.png")
            Image.fromarray(pred_mask).save(mask_path)

            
            desc_path = os.path.join(args.output_dir, "predictions", f"{base_name}_description.txt")
            with open(desc_path, 'w', encoding='utf-8') as f:
                f.write(description)

            print(f"Processed: {base_name}")

            
            if args.visualize:
                vis_path = os.path.join(args.output_dir, "visualizations", f"{base_name}_vis.png")
                predictor.visualize(image_path, pred_mask, save_path=vis_path)

        print(f"\nBatch processing complete! Successfully processed {successful}/{len(image_paths)} images")



    

    elif args.test_data_path and os.path.exists(args.test_data_path):

        

        print(f"Starting dataset evaluation: {args.test_data_path}")

        

        test_dataset = InferenceDataset(args.test_data_path)

        test_dataloader = DataLoader(

            test_dataset,

            batch_size=args.batch_size,

            shuffle=False,

            num_workers=args.num_workers,

            pin_memory=True,

        )

        print(f"Test sample count: {len(test_dataset)}")

        

        if len([gt for gt in test_dataset.gt_files if gt is not None]) > 0:

            print("\nStarting evaluation...")

            predictor.model.eval()

            metrics_accumulator = {

                : 0.0,

                : 0.0,

                : 0.0,

                : 0.0

            }

            count = 0

            with torch.no_grad():

                for batch_idx, (images, gts, images_ori, img_paths) in enumerate(

                        tqdm(test_dataloader, desc="Evaluation Progress")):

                    images = images.to(predictor.device)

                    gts = gts.to(predictor.device)

                    

                    for i in range(len(images)):

                        

                        single_image = images[i:i + 1]

                        single_gt = gts[i:i + 1]

                        img_path = img_paths[i]

                        

                        image = Image.open(img_path).convert('RGB')

                        

                        vlm_inputs = predictor.vlm_processor(image, return_tensors="pt").to(predictor.device)

                        vlm_outputs = predictor.vlm_model.generate(**vlm_inputs)

                        description = predictor.vlm_processor.decode(vlm_outputs[0], skip_special_tokens=True)

                        

                        mamba_inputs = predictor.tokenizer(description, padding=True, return_tensors="pt").to(
                            predictor.device)

                        mamba_outputs = predictor.mamba_model(**mamba_inputs)

                        

                        vision_outputs = predictor.vlm_model.vision_model(**vlm_inputs)

                        image_features = vision_outputs.last_hidden_state[:, 1:, :]

                        image_features = predictor.process_image_features(image_features)

                        

                        text_features = mamba_outputs.last_hidden_state

                        

                        pred = torch.sigmoid(predictor.model(single_image, text_features, image_features))

                        

                        pred_np = pred.squeeze().cpu().numpy()

                        gt_np = single_gt.squeeze().cpu().numpy()

                        

                        if pred_np.shape != gt_np.shape:
                            

                            import cv2

                            gt_np = cv2.resize(gt_np, (pred_np.shape[1], pred_np.shape[0]),
                                               interpolation=cv2.INTER_NEAREST)

                        

                        if pred_np.max() - pred_np.min() > 1e-7:

                            pred_norm = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-7)

                        else:

                            pred_norm = pred_np

                        

                        pred_tensor = torch.from_numpy(pred_norm).float()

                        gt_tensor = torch.from_numpy(gt_np).float()

                        

                        metrics = MetricsCalculator.compute_evaluation_all(pred_tensor, gt_tensor)

                        

                        for key in metrics_accumulator:
                            metrics_accumulator[key] += metrics[key]

                        count += 1

                        

                        if count % 1000 == 0:

                            avg_metrics = {k: v / count for k, v in metrics_accumulator.items()}

                            print(f"\nProgress: {count}/{len(test_dataset)}")

                            print("Current average metrics:")

                            for metric_name, value in avg_metrics.items():
                                print(f"  {metric_name}: {value:.4f}")

            

            if count > 0:

                avg_metrics = {k: v / count for k, v in metrics_accumulator.items()}

                print(f"\nEvaluation complete, processed {count} images")

                print("=" * 60)

                print("Evaluation Results:")

                print("=" * 60)

                for metric_name, value in avg_metrics.items():
                    print(f"{metric_name}: {value:.4f}")

                print("=" * 60)

                

                if args.save_results:

                    eval_result_path = os.path.join(args.output_dir, "evaluation_results.txt")

                    with open(eval_result_path, 'w', encoding='utf-8') as f:

                        f.write("VLSAM Evaluation Results\n")

                        f.write("=" * 40 + "\n")

                        f.write(f"Dataset: {args.test_data_path}\n")

                        f.write(f"Sample count: {count}\n")

                        f.write("=" * 40 + "\n")

                        for metric_name, value in avg_metrics.items():
                            f.write(f"{metric_name}: {value:.4f}\n")

                        f.write("=" * 40 + "\n")

                    print(f"Evaluation results saved to: {eval_result_path}")

        else:

            print("Warning: GT files not found, skipping evaluation")


if __name__ == "__main__":
    main()