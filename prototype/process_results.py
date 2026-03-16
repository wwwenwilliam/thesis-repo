"""
Standalone script to process similarity join results.

Usage:
  python process_results.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util.process_results import process_results, dump_statistics

RESULTS_DIR = "./results/10M"


def main():
    stats = process_results(RESULTS_DIR)
    dump_statistics(RESULTS_DIR, stats)


if __name__ == "__main__":
    main()
