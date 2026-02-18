# /disk4/apardini-gdpalma-paper/setup.py

from setuptools import setup, find_packages

setup(
    name="utils",
    version="0.1.0",
    description="Pacchetto condiviso di utilità per i progetti Pardini-DiPalma.",
    author="apardini-gdpalma",
    packages=find_packages(include=['Utils', 'Utils.*']),
)