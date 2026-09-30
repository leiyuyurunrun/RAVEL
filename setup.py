from setuptools import setup, find_packages

setup(
    name="evotx",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "anthropic>=0.112.0",
        "openai>=1.0.0",
        "python-dotenv>=1.0.0",
    ],
)
