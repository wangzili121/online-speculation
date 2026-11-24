"""Bind the chip entry of standalone PyPTO programs to the resident ABI."""

from pathlib import Path


def with_resident_entry(host, chip):
    host.resident_entry = chip
    return host


def resident_entry(program):
    return getattr(program, "resident_entry", program)


def compile_resident_entry(program, arguments, config, *, abi, name):
    compiled = program.compile(*arguments, config=config)
    if not (Path(compiled.output_dir) / "compiled_meta.json").is_file():
        raise TypeError(f"{name}: resident execution requires a PyPTO L2 chip entry")
    return compiled, abi(compiled)
