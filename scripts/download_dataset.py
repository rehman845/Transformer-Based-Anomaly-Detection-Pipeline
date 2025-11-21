import argparse
import zipfile
from pathlib import Path
from typing import Dict

import requests
from tqdm import tqdm


DATASET_URLS: Dict[str, str] = {
    "smd": "https://github.com/elisejiuqizhang/TS-AD-Datasets/raw/main/datasets/SMD.zip",
    "msl": "https://github.com/elisejiuqizhang/TS-AD-Datasets/raw/main/datasets/MSL.zip",
    "smap": "https://github.com/elisejiuqizhang/TS-AD-Datasets/raw/main/datasets/SMAP.zip",
    "swat": "https://github.com/elisejiuqizhang/TS-AD-Datasets/raw/main/datasets/SWaT.zip",
}


def download_file(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {url}") as pbar:
            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)


def main(name: str, output_dir: str) -> None:
    name = name.lower()
    if name not in DATASET_URLS:
        raise ValueError(f"Unknown dataset {name}. Available: {list(DATASET_URLS)}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    zip_path = output_root / f"{name}.zip"
    if not zip_path.exists():
        download_file(DATASET_URLS[name], zip_path)
    else:
        print(f"Zip file {zip_path} already exists. Skipping download.")

    dataset_dir = output_root / name.upper()
    if dataset_dir.exists():
        print(f"Dataset folder {dataset_dir} already exists. Skipping extraction.")
        return

    extract_zip(zip_path, output_root)
    print(f"Extracted dataset to {dataset_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download TS-AD datasets")
    parser.add_argument("--name", type=str, required=True, help="Dataset name (smd, msl, smap, swat)")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Directory to store data")
    args = parser.parse_args()

    main(args.name, args.output_dir)

