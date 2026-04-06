from __future__ import annotations

import argparse
from pathlib import Path

import torch
from mmseg.apis import init_model


class SegWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model.encode_decode(x, batch_img_metas=[dict(ori_shape=x.shape[-2:], img_shape=x.shape[-2:], pad_shape=x.shape[-2:])])
        return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MMSeg model to ONNX")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shape", nargs=2, type=int, default=[512, 512], metavar=("H", "W"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    model = init_model(args.config, args.checkpoint, device="cpu")
    model.eval()
    wrapper = SegWrapper(model).eval()

    h, w = args.shape
    dummy = torch.randn(1, 3, h, w, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy,
        str(output),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
    )

    print(f"onnx exported to {output}")


if __name__ == "__main__":
    main()
