
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc


def print_memory_stats(label=""):
    allocated = torch.cuda.memory_allocated() / 1e9
    cached = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9
    print(f"{label}: Allocated={allocated:.2f}GB, Cached={cached:.2f}GB, Max Allocated={max_allocated:.2f}GB")


def test_sam_baseline():
    
    from segment_anything import sam_model_registry

    print("=== Test Base SAM Model ===")
    torch.cuda.empty_cache()

    model = sam_model_registry["vit_l"](checkpoint="/root/autodl-tmp/MMsam/sam/sam_vit_l_0b3195.pth").to("cuda:0")
    model.eval()

    print_memory_stats("After model loading")

    
    x = torch.randn(1, 3, 1024, 1024).cuda()
    print_memory_stats("After creating input")

    with torch.no_grad():
        image_embedding = model.image_encoder(x)
        print_memory_stats("After image encoder")

        
        output = model(x, multimask_output=False)
        print_memory_stats("After full inference")

    return model


def test_integrated_step_by_step():
    
    print("\n=== Step-by-step test of integrated model ===")

    
    from segment_anything.modeling.mcsam_integrated import create_integrated_model

    
    torch.cuda.empty_cache()
    gc.collect()

    
    model = create_integrated_model(
        sam_checkpoint_path="/root/autodl-tmp/MMsam/sam/sam_vit_l_0b3195.pth",
        model_type="vit_l",
        image_size=1024,
        use_rankdice=True,
        use_hypercond=True,
        n_streams=4,
        device="cuda:0"
    ).to("cuda:0")

    model.eval()
    print_memory_stats("After integrated model loading")

    
    batch_size = 1

    
    image = torch.randn(batch_size, 3, 1024, 1024).cuda()
    print_memory_stats("After creating image input")

    
    text_embeddings = torch.randn(batch_size, 77, 768).cuda()
    print_memory_stats("After creating text features")

    
    image_features = torch.randn(batch_size, 576, 768).cuda()  
    print_memory_stats("After creating image features")

    
    print("\n=== Test Image Encoder ===")
    with torch.no_grad():
        
        print_memory_stats("Before start")
        image_embedding = model.image_encoder(image, image_features)
        print_memory_stats("After image encoder")
        print(f"Image embedding shape: {image_embedding.shape}")

        
        print("\n=== Test Prompt Generator ===")
        text_global = text_embeddings.mean(dim=1)
        vision_global = image_features.mean(dim=1)

        sparse_prompt = model.prompt_generator(
            text_global,
            vision_global,
            text_embeddings,
            image_features
        )
        print_memory_stats("After prompt generator")
        print(f"Sparse prompt shape: {sparse_prompt.shape}")

        
        print("\n=== Test Hyperparameter Conditioning ===")
        if model.use_hypercond:
            cond_embedding = model.hyper_cond(0.5, 1.0, "cuda:0")
            print_memory_stats("After hyperparameter conditioning")
            print(f"Condition embedding shape: {cond_embedding.shape}")

        
        print("\n=== Full Forward Pass ===")
        model.eval()
        pred_mask, rank_dice_loss, hyper_cond = model(
            image=image,
            text_embeddings=text_embeddings,
            image_features=image_features,
            gt_mask=None,
            hyper_cond={'threshold': 0.5, 'boundary_weight': 1.0},
            return_logits=False
        )
        print_memory_stats("After full forward")
        print(f"Predicted mask shape: {pred_mask.shape}")


def test_attention_memory():
    
    print("\n=== Test Attention Memory ===")

    
    image_size = 1024
    patch_size = 16
    seq_len = (image_size // patch_size) ** 2  
    num_heads = 16
    batch_size = 1

    
    attn_size = batch_size * num_heads * seq_len * seq_len

    
    attn_memory_gb = attn_size * 4 / 1e9

    print(f"Sequence length: {seq_len}")
    print(f"Attention heads: {num_heads}")
    print(f"Attention matrix elements: {attn_size:,}")
    print(f"Attention matrix memory (float32): {attn_memory_gb:.2f} GB")
    print(f"Attention matrix memory (bfloat16): {attn_memory_gb / 2:.2f} GB")

    
    torch.cuda.empty_cache()
    print_memory_stats("Before start")

    try:
        
        attn_matrix = torch.randn(batch_size, num_heads, seq_len, seq_len).cuda()
        print_memory_stats("After allocating attention matrix")
        print("✅ Full attention matrix can be allocated")
    except Exception as e:
        print(f"❌ Unable to allocate full attention matrix: {e}")


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    
    test_attention_memory()

    
    

    
    test_integrated_step_by_step()

    print("\n=== Memory Test Complete ===")