from setuptools import setup, find_packages

setup(
    name="florence2-client",
    version="0.1.0",
    description="Florence-2 Client and Server Package",
    packages=find_packages(include=["florence2_client", "florence2_server", "florence2_client.*", "florence2_server.*"]),
    install_requires=[
        "fastapi",
        "uvicorn",
        "requests",
        "pillow",
        "transformers",
        "torch",
        "supervision"
    ],
    extras_require={
        "client": ["requests", "supervision", "pillow"],
        "server": ["fastapi", "uvicorn", "pillow", "transformers", "torch"],
    },
    python_requires=">=3.10"
)