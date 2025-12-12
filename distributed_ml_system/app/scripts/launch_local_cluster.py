"""Launch a local Ray cluster with optional GPU simulation."""
from __future__ import annotations

import argparse
import ray


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs to allocate")
    parser.add_argument("--num-cpus", type=int, default=4, help="Number of CPUs to allocate")
    args = parser.parse_args()

    ray.init(num_gpus=args.num_gpus, num_cpus=args.num_cpus)
    print(ray.cluster_resources())


if __name__ == "__main__":
    main()
