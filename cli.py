"""Command-line entry point."""
import argparse
import json

from .inference import predict
from .modeling import train_and_evaluate


def main():
    parser = argparse.ArgumentParser(description="Reproducible ML portfolio pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and evaluate the model")
    train_parser.add_argument("--output-dir", default="artifacts")
    train_parser.add_argument("--seed", type=int, default=42)

    predict_parser = subparsers.add_parser("predict", help="Run inference on one feature row")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument(
        "--values", required=True,
        help="Six comma-separated numeric feature values"
    )

    args = parser.parse_args()
    if args.command == "train":
        result = train_and_evaluate(args.output_dir, args.seed)
        print(json.dumps(result, indent=2))
    elif args.command == "predict":
        row = [float(value.strip()) for value in args.values.split(",")]
        print(json.dumps(predict(args.model, [row]), indent=2))


if __name__ == "__main__":
    main()
