import subprocess


def main(key_len):
    cmd = [
        "python",
        "train_LDM.py",
        "--gpu_id=5",
        "--dataset_name=ffhq",
        "--dataloader_num_workers=8",
        "--wandb_mode=online",
        "--resolution=256",
        "--center_crop",
        "--random_flip",
        "--lpips_reg",
        "--checkpointing_steps=10000",
        "--mixed_precision=no",
        "--max_train_steps=50_000",
        "--train_batch_size=32",
        "--gradient_accumulation_steps=4",
        "--learning_rate=1e-4",
        "--lr_scheduler=cosine_with_restarts",
        "--cosine_cycle=1000",
        "--lr_warmup_steps=100",
        "--decoding_network_type=resnet",
        "--finetune=all",
        "--modulation=de",
        "--mapping_network_type=sg2",
        "--mapping_layer=2",
        f"--phi_dimension={key_len}",
        "--int_dimension=128",
        "--attack=""",
        "--output_dir=checkpoints/ffhq/32/",
        "--vae_config=configs/first_stage_models/vq-f4/model.yaml"
    ]

    print(cmd)
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    key_dim = [32]

    for dim in key_dim:
        main(dim)
