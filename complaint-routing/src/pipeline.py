"""Run the data pipeline end to end: extract, then clean and split."""

import logging

import clean
import extract


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    extract.main()
    clean.main()


if __name__ == "__main__":
    main()