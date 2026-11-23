#!/usr/bin/env python3
"""Fail fast unless the DSV4.1 resident online factory is production complete."""

import json

from blockspec.pypto_dsv41_resident_primitives import production_readiness


def main():
    report = production_readiness()
    print(json.dumps(report, indent=2))
    if not report["production_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
