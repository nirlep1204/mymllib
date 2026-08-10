import sys
import re
from .help import help as ml_help


def main():
    args = sys.argv[1:]
    if not args:
        ml_help()
        return

    full_cmd = " ".join(args).strip()

    # Match patterns like help(func_name) or --help(func_name)
    m = re.search(r"--?help\(([^)]+)\)|help\(([^)]+)\)", full_cmd)
    if m:
        target = m.group(1) or m.group(2)
        ml_help(target.strip("'\" "))
        return

    # Match patterns like --help=func_name
    if "--help=" in full_cmd:
        target = full_cmd.split("--help=", 1)[1].strip().strip("'\" ")
        ml_help(target)
        return

    # Filter flags and strip quotes
    remaining = [a.strip("'\" ") for a in args if a not in ("--help", "-h", "help", "--")]
    if remaining:
        ml_help(remaining[0])
    else:
        ml_help()


if __name__ == "__main__":
    main()
