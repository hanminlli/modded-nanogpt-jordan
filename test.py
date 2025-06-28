
import os

if __name__ == "__main__":
    rank = int(os.environ["RANK"]) # global rank of the process
    world_size = int(os.environ["WORLD_SIZE"])

    print(rank, world_size)