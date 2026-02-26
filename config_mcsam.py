


import os
from dataclasses import dataclass


@dataclass
class IntegratedModelConfig:
    
    
    model_type: str = "vit_l"
    image_size: int = 1024
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12

    
    use_rankdice: bool = True
    use_hypercond: bool = True
    n_streams: int = 4
    prompt_dim: int = 256
    cond_dim: int = 128

    
    threshold_range: tuple = (0.3, 0.7)
    boundary_weight_range: tuple = (0.5, 2.0)

    
    batch_size: int = 1
    num_epochs: int = 20
    learning_rate: float = 0.0002
    weight_decay: float = 0.01
    warmup_steps: int = 100

    
    work_dir: str = "./work_dir"
    model_save_path: str = "./checkpoints"
    log_dir: str = "./logs"

    def __post_init__(self):
        
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    @classmethod
    def from_args(cls, args):
        
        return cls(
            model_type=args.model_type,
            use_rankdice=args.use_rankdice,
            use_hypercond=args.use_hypercond,
            n_streams=args.n_streams,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            work_dir=args.work_dir,
        )



default_config = IntegratedModelConfig()