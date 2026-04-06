from __future__ import annotations

import argparse

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test an MMSeg model with local runner")
    parser.add_argument("--config", required=True, help="config path")
    parser.add_argument("--checkpoint", required=True, help="checkpoint path")
    parser.add_argument("--work-dir", default=None, help="override work dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    if args.work_dir:
        cfg.work_dir = args.work_dir
    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == "__main__":
    main()
