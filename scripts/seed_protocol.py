"""打印或执行论文规定的 miner/sampling/downstream 三阶段种子协议。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# 同时支持 ``python scripts/seed_protocol.py`` 与测试中的模块化加载。
_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(_SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIRECTORY))

import _bootstrap  # noqa: F401

from nam.config import load_config


MINER_SEED = 42
DOWNSTREAM_SEED = 42
SAMPLING_SEEDS = (42, 3407, 2026)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/table1_2d.yaml")
    parser.add_argument("--spatial-dims", type=int, choices=(2, 3), default=2)
    parser.add_argument("--method", choices=("base", "nam"), default="nam")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="按打印顺序实际执行；默认仅打印，便于提交到服务器任务系统。",
    )
    return parser.parse_args()


def _stage_overrides(sampling_seed: int) -> list[str]:
    return [
        f"runtime.miner_seed={MINER_SEED}",
        f"runtime.downstream_seed={DOWNSTREAM_SEED}",
        f"runtime.sampling_seed={sampling_seed}",
    ]


def build_commands(config_path: str, spatial_dims: int, method: str) -> list[list[str]]:
    """构造无歧义命令；三组采样及其下游训练严格一一对应。"""
    config = load_config(config_path)
    diffusion_name = str(config.diffusion.name).lower()
    method_section = getattr(config, diffusion_name, None)
    settings = (
        method_section.sampling
        if method_section is not None and hasattr(method_section, "sampling")
        else config.sampling
    )
    config_argument = Path(config_path).as_posix()
    script_suffix = f"{spatial_dims}d"
    commands: list[list[str]] = [
        [
            "python",
            f"scripts/train_nam_{script_suffix}.py",
            "--config",
            config_argument,
            "--set",
            *_stage_overrides(SAMPLING_SEEDS[0]),
        ]
    ]
    for sampling_seed in SAMPLING_SEEDS:
        overrides = _stage_overrides(sampling_seed)
        synthetic_root = (
            Path(str(settings.output_dir))
            / f"seed_{sampling_seed}"
            / f"{config.experiment_name}-{method}"
        ).as_posix()
        commands.append(
            [
                "python",
                f"scripts/generate_{script_suffix}.py",
                "--config",
                config_argument,
                "--method",
                method,
                "--set",
                *overrides,
            ]
        )
        commands.append(
            [
                "python",
                f"scripts/train_downstream_{script_suffix}.py",
                "--config",
                config_argument,
                "--phase",
                "synthetic",
                "--set",
                *overrides,
                f"synthetic_dataset.root={synthetic_root}",
            ]
        )
    return commands


def main() -> None:
    arguments = parse_args()
    commands = build_commands(arguments.config, arguments.spatial_dims, arguments.method)
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {subprocess.list2cmdline(command)}", flush=True)
        if arguments.execute:
            # 使用当前解释器执行，但打印命令保留论文/README 中统一的 python 写法。
            executable_command = [sys.executable, *command[1:]]
            subprocess.run(executable_command, check=True)


if __name__ == "__main__":
    main()
