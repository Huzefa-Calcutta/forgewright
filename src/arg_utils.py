import argparse
from pathlib import Path
from typing import Any, Dict


def parse_cli_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forgewright CNC tool-wear analysis: per-job power/vibration metrics "
                    "and two-stage anomaly detection."
    )
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"),
                        help="Path to the YAML config file.")
    parser.add_argument("--input-dir", default=None, help="Override paths.input_dir.")
    parser.add_argument("--output-dir", default=None, help="Override paths.output_dir.")
    parser.add_argument("--model-dir", default=None, help="Override paths.model_dir.")
    parser.add_argument("--primary-model", default=None, choices=["ridge", "huber", "ransac", "robust"],
                        help="Override primary.model.")
    parser.add_argument("--secondary-model", default=None,
                        choices=["isolation_forest", "mahalanobis", "lof"],
                        help="Override secondary.model.")
    parser.add_argument("--mode", default=None, choices=["fit", "load"], help="Override run.mode.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging.")
    return parser.parse_args(argv)


def cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {
        "paths": {"input_dir": args.input_dir, "output_dir": args.output_dir, "model_dir": args.model_dir},
        "primary": {"model": args.primary_model},
        "secondary": {"model": args.secondary_model},
        "run": {"mode": args.mode, "verbose": False if args.quiet else None},
    }
    return overrides