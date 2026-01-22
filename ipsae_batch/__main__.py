"""
Entry point for running ipsae_batch as a module.

Usage:
    python -m ipsae_batch <input_folder> [options]
"""

from .cli import main

if __name__ == "__main__":
    main()
