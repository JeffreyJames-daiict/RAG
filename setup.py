from setuptools import setup, find_packages

setup(
    name="enterprise-rag",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pydantic>=2.0.0",
        "sqlalchemy>=2.0.0",
        "boto3>=1.30.0",
        "requests>=2.28.0",
        "aiohttp>=3.8.0",
    ],
)
