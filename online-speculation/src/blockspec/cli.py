"""Installed entry points for the retained drafting and continuation pipeline."""

import argparse
import importlib
import sys


COMMANDS = {"evaluate": "evaluate", "online": "continue_training", "fit": "fit"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=tuple(COMMANDS))
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    module = importlib.import_module(f"blockspec.commands.{COMMANDS[args.command]}")
    sys.argv = [sys.argv[0], *remaining]
    module.main()
