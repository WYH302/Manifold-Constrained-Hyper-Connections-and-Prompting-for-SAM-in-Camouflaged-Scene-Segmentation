































































































































































































import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys


def test_training_mode():
    
    print("=== Test Training Mode ===")

    
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from segment_anything.modeling.mcsam_integrated import create_integrated_model

    device = torch.device("cuda:0")

    
    model = create_integrated_model(
        sam_checkpoint_path="/root/autodl-tmp/MMsam/sam/sam_vit_l_0b3195.pth",
        model_type="vit_l",
        image_size=1024,
        use_rankdice=True,
        use_hypercond=True,
        n_streams=4,
        device=device,
        blip_feature_dim=768
    ).to(device)

    
    model.train()
    print(f"Training mode: {model.training}")

    
    batch_size = 1
    image = torch.randn(batch_size, 3, 1024, 1024).to(device)
    text_embeddings = torch.randn(batch_size, 77, 768).to(device)
    image_features = torch.randn(batch_size, 576, 768).to(device)
    gt_mask = torch.randn(batch_size, 1, 1024, 1024).to(device)

    
    torch.cuda.empty_cache()
    print(f"Initial memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

    
    print("\nTesting training forward pass...")
    logits, rank_dice_loss, hyper_cond = model(
        image=image,
        text_embeddings=text_embeddings,
        image_features=image_features,
        gt_mask=gt_mask,
        hyper_cond={'threshold': 0.5, 'boundary_weight': 1.0},
        return_logits=True
    )

    print(f"Memory after training forward: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print(f"Max memory: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
    print(f"logits shape: {logits.shape}, Loss: {rank_dice_loss}")

    
    print("\nTesting backpropagation...")
    total_loss = rank_dice_loss + F.binary_cross_entropy_with_logits(logits, gt_mask)

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
    optimizer.zero_grad()

    total_loss.backward()
    optimizer.step()

    print(f"Memory after backpropagation: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print(f"Max memory: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")

    return model


def test_amp_training():
    
    print("\n=== Test Mixed Precision Training ===")

    from segment_anything.modeling.mcsam_integrated import create_integrated_model

    device = torch.device("cuda:0")

    
    model = create_integrated_model(
        sam_checkpoint_path="/root/autodl-tmp/MMsam/sam/sam_vit_l_0b3195.pth",
        model_type="vit_l",
        image_size=1024,
        use_rankdice=True,
        use_hypercond=True,
        n_streams=4,
        device=device,
        blip_feature_dim=768
    ).to(device)

    model.train()

    
    batch_size = 1
    image = torch.randn(batch_size, 3, 1024, 1024).to(device)
    text_embeddings = torch.randn(batch_size, 77, 768).to(device)
    image_features = torch.randn(batch_size, 576, 768).to(device)
    gt_mask = torch.randn(batch_size, 1, 1024, 1024).to(device)

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
    scaler = torch.cuda.amp.GradScaler()

    
    torch.cuda.empty_cache()
    print(f"Initial memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

    
    print("Training with AMP...")
    optimizer.zero_grad()

    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        logits, rank_dice_loss, hyper_cond = model(
            image=image,
            text_embeddings=text_embeddings,
            image_features=image_features,
            gt_mask=gt_mask,
            hyper_cond={'threshold': 0.5, 'boundary_weight': 1.0},
            return_logits=True
        )

        total_loss = rank_dice_loss + F.binary_cross_entropy_with_logits(logits, gt_mask)

    print(f"Memory after forward: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print(f"Memory after backpropagation: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print(f"Max memory: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")

    return model


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")

    
    print("\n" + "=" * 50)
    model1 = test_training_mode()

    
    del model1
    torch.cuda.empty_cache()

    
    print("\n" + "=" * 50)
    model2 = test_amp_training()


if __name__ == "__main__":
    main()