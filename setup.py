import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "MLOPS_Project_Skin_Cancer_Detection"
AUTHOR_USER_NAME = "DA23S001"
SRC_REPO = "Skin_Cancer_classifier"
AUTHOR_EMAIL = "DA23S001@smail.iitm.ac.in"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small package for CNN app",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    package_dir = {"": "src"},
    packages= setuptools.find_packages(where="src")
)