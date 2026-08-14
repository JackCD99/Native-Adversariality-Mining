"""List the complete Table I experiment matrix and executable base configs."""

import argparse
import itertools
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print Table I experiment cells.")
    parser.add_argument("--matrix", default="configs/table1_matrix.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    with Path(arguments.matrix).open("r", encoding="utf-8") as stream:
        matrix = yaml.safe_load(stream)
    count = 0
    for dataset_name, dataset in matrix["datasets"].items():
        dimension = f"{dataset['spatial_dims']}d"
        for diffusion, downstream, seed in itertools.product(
            dataset["diffusion_models"], matrix["downstream_models"][dimension], matrix["seeds"]
        ):
            count += 1
            pipeline = matrix["pipelines"][diffusion]
            if int(pipeline["spatial_dims"]) != int(dataset["spatial_dims"]):
                raise ValueError(f"Dimension mismatch for {dataset_name}/{diffusion}.")
            print(
                f"{count:03d} dataset={dataset_name} diffusion={diffusion} "
                f"downstream={downstream} seed={seed} config={pipeline['config']}"
            )
    print(f"Total experiment cells: {count}")
