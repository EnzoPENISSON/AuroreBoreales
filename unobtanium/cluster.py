import os
import argparse
import torch.distributed as dist
from predict import train_model

def run_distributed_training(rank, world_size, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['NODE_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = '0'

    # Critical: force Gloo to use the right network interface
    # Run ip addr and replace 'eth0' with your actual LAN interface
    os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'

    print(f"--- Node Rank {rank} of {world_size} ---")
    print(f"--- Master: {master_addr}:{master_port} ---")
    train_model(retrain=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--master_addr", type=str, default="192.168.1.120")
    parser.add_argument("--master_port", type=int, default=12355)
    args = parser.parse_args()
    run_distributed_training(args.rank, 2, args.master_addr, args.master_port)