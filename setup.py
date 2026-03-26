"""Backward-compatible setup.py for fathom-depth.

All configuration is in pyproject.toml. This file exists only so that
`pip install -e .` works on older pip versions that don't read
pyproject.toml natively.
"""
from setuptools import setup

if __name__ == "__main__":
    setup()
