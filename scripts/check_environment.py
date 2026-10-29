#!/usr/bin/env python3
"""Repository entry point; implementation ships in the installed package."""
from inventory_di.doctor import main, run_environment_doctor

__all__ = ["run_environment_doctor"]

if __name__ == "__main__":
    raise SystemExit(main())
