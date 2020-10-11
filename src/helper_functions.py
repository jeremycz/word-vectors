import os
from pathlib import Path
import requests
import gzip
import shutil
import re
import string


def get_repo_root() -> Path:
    # Check for dataset folder and adjust path if necessary
    repo_root = Path(os.getcwd())
    if not os.path.exists(repo_root / "data"):
        repo_root = repo_root.parent

    return repo_root


def download_dataset(dataset_url: str) -> str:
    repo_root = get_repo_root()

    print(f"Root folder: {repo_root}")

    dataset_folder = repo_root / "data"
    dataset_path = Path(dataset_folder / "books.gz")
    extracted_dataset_path = Path(dataset_folder / "books.txt")

    if not os.path.exists(extracted_dataset_path):
        if not os.path.exists(dataset_path):
            print(f"Downloading dataset from {dataset_url}")
            req = requests.get(dataset_url, allow_redirects=True)
            with open(dataset_path, "wb") as f:
                f.write(req.content)

            print("Extracting dataset")
            with gzip.open(dataset_path, "rb") as f_in:
                with open(extracted_dataset_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
    else:
        print(f"Extracted dataset found at {extracted_dataset_path}")

    return extracted_dataset_path


def clean_sentence(sentence: str) -> str:
    # Lowercase
    sentence = sentence.lower()

    # Skip if sentence contains URL
    if sentence.find("http") >= 0:
        return str()

    # Remove punctuation
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))

    # Remove consecutive spaces
    sentence = re.sub(r"\s\s+", " ", sentence)

    # Strip leading/trailing whitespace and newlines
    sentence = sentence.strip()

    return sentence