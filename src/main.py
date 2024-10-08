from argparse import ArgumentParser, ArgumentError
from detection import dataset_checker, inference, train


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        parents=[
            dataset_checker.build_parser(),
            inference.build_parser(),
            train.build_parser()
            ],
        add_help=True
    )

    parser.add_argument(
        "-m", "--module", type=str, 
        help="Module name to run"    
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.module == "inference":
        inference.run(args)
    elif args.module == "dataset_checker":
        dataset_checker.run(args)
    elif args.modulde == "train":
        train.run(args)
    else:
        raise ArgumentError(f"No such available module {args.module}")


if __name__ == "__main__":
    main()
