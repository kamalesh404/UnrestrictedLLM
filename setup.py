from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="unrestrictedllm",
    version="0.1.0",
    description="Run uncensored open-source LLMs locally or on cloud",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kamalesh",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "click>=8.0",
        "fastapi>=0.100",
        "uvicorn>=0.20",
        "pydantic>=2.0",
        "requests>=2.28",
        "huggingface_hub>=0.16",
    ],
    extras_require={
        "llama_cpp": ["llama-cpp-python>=0.2"],
        "transformers": ["transformers>=4.30", "torch>=2.0"],
        "cloud": ["modal", "runpod"],
        "ui": ["gradio>=3.40"],
        "dev": ["pytest>=7.0", "ruff", "mypy", "pytest-asyncio"],
    },
    entry_points={
        "console_scripts": [
            "unrestricted=src.cli.main:cli",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
