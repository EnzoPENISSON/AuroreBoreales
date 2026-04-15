import os
import argparse
import torch.distributed as dist
from predict import train_model


def run_distributed_training(rank, world_size, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    # ← AJOUTER CES LIGNES :
    os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'  # Sur Linux, remplacer par votre interface réseau
    # ou 'en0' sur macOS, etc.

    print(f"--- Initializing Node Rank {rank} of {world_size} ---")
    print(f"--- Connecting to Master at {master_addr}:{master_port} ---")

    train_model(retrain=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-node Distributed Trainer")
    parser.add_argument("--rank", type=int, required=True, help="Rank of this PC (0 for Master, 1 for Worker)")
    parser.add_argument("--master_addr", type=str, default="192.168.1.10", help="IP address of PC 0")
    parser.add_argument("--master_port", type=int, default=12355, help="Free port on PC 0")

    args = parser.parse_args()

    # world_size is 2 as per your requirement
    run_distributed_training(args.rank, 2, args.master_addr, args.master_port)