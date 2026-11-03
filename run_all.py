"""Run the whole workflow end to end.

    python run_all.py              # EDA + regression + classification
    python run_all.py --skip-eda   # just train the models
"""
import argparse

from anxiety import eda, train_classification, train_regression


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--skip-eda", action="store_true", help="skip the exploratory analysis step")
    parser.add_argument("--skip-classification", action="store_true")
    args = parser.parse_args()

    if not args.skip_eda:
        print("=" * 30, "EDA", "=" * 30)
        eda.main()
    print("=" * 26, "REGRESSION", "=" * 26)
    train_regression.main()
    if not args.skip_classification:
        print("=" * 24, "CLASSIFICATION", "=" * 24)
        train_classification.main()


if __name__ == "__main__":
    main()
