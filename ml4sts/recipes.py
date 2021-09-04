# Imports: standard library
import sys
import logging
from timeit import default_timer as timer

# Imports: ml4sts
# Imports: first party
from ml4sts.train import train
from ml4sts.compare import compare_across, compare_within
from ml4sts.predict import predict, predict_ffs
from ml4sts.arguments import parse_args
from ml4sts.summary_statistics import summary_statistics
from ml4sts.sensitivity_analysis import sensitivity_analysis
from ml4sts.forward_feature_selection import ffs


def run(args):
    start_time = timer()

    try:
        if args.mode == "summary_statistics":
            summary_statistics(args)
        elif args.mode == "train":
            train(args)
        elif args.mode == "ffs":
            ffs(args)
        elif args.mode == "compare_across":
            compare_across(args)
        elif args.mode == "compare_within":
            compare_within(args)
        elif args.mode == "sensitivity_analysis":
            sensitivity_analysis(args)
        elif args.mode == "predict":
            predict(args)
        elif args.mode == "predict_ffs":
            predict_ffs(args)
        else:
            raise ValueError("Unknown mode:", args.mode)
    except Exception as err:
        logging.exception(err)

    end_time = timer()
    elapsed_time = end_time - start_time
    logging.info(f"Executed {args.mode} mode in {elapsed_time:.2f} sec")


def main():
    args = parse_args(sys.argv[1:])
    if args.mode:
        run(args)


if __name__ == "__main__":
    main()
