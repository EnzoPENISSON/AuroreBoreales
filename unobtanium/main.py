import argparse
import os
import torch
import torch.distributed as dist
from predict import train_model, create_model

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def setup(rank: int, world_size: int, master_ip: str, master_port: int):
    os.environ["MASTER_ADDR"] = master_ip
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)  # needed by Lightning

    device = get_device()
    print(f"[rank {rank}/{world_size}] using device: {device}")

    backend = "nccl" if device.type == "cuda" else "gloo"
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size
    )

def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Existing arguments
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=2)  # Changed to hyphen to match command
    parser.add_argument("--master_addr", type=str, default="192.168.1.120")
    parser.add_argument("--master_port", type=int, default=12355)
    parser.add_argument("--retrain", action="store_true")

    # New arguments from your bash command
    parser.add_argument("--dist-url", type=str, default="tcp://127.0.0.1:12355")
    parser.add_argument("--dist-backend", type=str, default="gloo")
    parser.add_argument("--dummy", action="store_true", help="Use dummy data")
    parser.add_argument("--no-accel", action="store_true", help="Disable accelerator")
    parser.add_argument("-a", "--arch", metavar="ARCH", default="resnet18")

    args = parser.parse_args()

    try:
        setup(args.rank, args.world_size, args.master_addr, args.master_port)
        train_model(retrain=args.retrain)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
