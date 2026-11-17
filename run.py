"""Entry point: python run.py --config config.json"""
import argparse
from pathlib import Path

from two_wcf.config import load_config
from two_wcf.pipeline import run


def main():
    parser = argparse.ArgumentParser(description='2WCF manual Farneback, one TIFF sequence')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'),
                        help='Path to a JSON configuration file (default: alongside run.py)')
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == '__main__':
    main()
