"""Developer-friendly offline preparation commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from .calibration import FP8Format, calibrate_fp8
from .model import FixedShapeTransactionTransformer, TransactionTransformerConfig
from .sample import make_synthetic_transactions


def main(argv: Sequence[str] | None = None) -> int:
    """Run deterministic synthetic generation or calibration from the CLI."""
    parser = argparse.ArgumentParser(description="Offline fraud-model preparation utilities.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    sample_parser = subcommands.add_parser("sample", help="print deterministic synthetic batch details")
    sample_parser.add_argument("--batch-size", type=int, default=8)
    sample_parser.add_argument("--seed", type=int, default=7)
    calibration_parser = subcommands.add_parser("calibrate", help="emit FP8 activation scale metadata")
    calibration_parser.add_argument("output", type=Path)
    calibration_parser.add_argument("--format", choices=[item.value for item in FP8Format], default="E4M3")
    calibration_parser.add_argument("--batch-size", type=int, default=8)
    calibration_parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    config = TransactionTransformerConfig()
    torch.manual_seed(args.seed)
    batch = make_synthetic_transactions(config, batch_size=args.batch_size, seed=args.seed)
    if args.command == "sample":
        print(json.dumps({"features_shape": list(batch.features.shape), "labels": batch.labels.flatten().tolist()}))
        return 0

    result = calibrate_fp8(
        FixedShapeTransactionTransformer(config), [batch.features], FP8Format(args.format)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as output_file:
        json.dump(result.to_dict(), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
