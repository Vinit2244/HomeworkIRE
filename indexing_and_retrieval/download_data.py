# ======================== IMPORTS ========================
import os
import utils
import argparse
import subprocess
from utils import style


# ======================= FUNCTIONS =======================
def download_news_dataset(destination: str) -> None:
    print(f"{style.FG_CYAN}Downloading news dataset...{style.RESET}")

    # Make sure the destination directory exists
    os.makedirs(destination, exist_ok=True)

    # Clone the repository
    repo_url = "https://github.com/Webhose/free-news-datasets.git"
    subprocess.run(["git", "clone", repo_url, destination], check=True)

    print(f"{style.FG_GREEN}News dataset downloaded at {destination}\n{style.RESET}")


def download_wikipedia_dataset() -> None:
    print(f"{style.FG_CYAN}Use wikipedia dataset directly using datasets library.\n{style.RESET}")


def unzip_and_clean_news_dataset(path_to_news_dataset: str) -> None:
    print(f"{style.FG_CYAN}Unzipping and cleaning news dataset...{style.RESET}")

    # Remove all the files except the "News_Datasets" folder
    all_items: list = os.listdir(path_to_news_dataset)
    for item in all_items:
        if item != "News_Datasets":
            item_path: str = os.path.join(path_to_news_dataset, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                subprocess.run(["rm", "-rf", item_path], check=True)
    
    # Unzip all the zipped folders in "News_Datasets" and remove the zipped files
    path_to_zipped_folders: str = os.path.join(path_to_news_dataset, "News_Datasets")
    all_zipped_folders: list = os.listdir(path_to_zipped_folders)
    unzipped_count = 0
    for folder in all_zipped_folders:
        folder_path: str = os.path.join(path_to_zipped_folders, folder)
        if folder_path.endswith(".zip"):
            try:
                subprocess.run(["unzip", "-o", folder_path, "-d", folder_path[:-4]], check=True)
                unzipped_count += 1
                os.remove(folder_path)
            except:
                subprocess.run(["rm", "-rf", folder_path], check=True)
    print(f"{style.FG_GREEN}Unzipped {unzipped_count}/{len(all_zipped_folders)}{style.RESET}")

    # Move all the unzipped folders to the main directory and remove the empty folders
    all_unzipped_folders: list = [f for f in os.listdir(path_to_zipped_folders) if not f.endswith(".zip")]
    for folder in all_unzipped_folders:
        folder_path: str = os.path.join(path_to_zipped_folders, folder)
        subprocess.run(["mv", folder_path, path_to_news_dataset], check=True)
    subprocess.run(["rm", "-rf", path_to_zipped_folders], check=True)

    print(f"{style.FG_GREEN}News dataset unzipped and cleaned at {path_to_zipped_folders}\n{style.RESET}")


# ======================== MAIN ==========================
def main(args) -> None:
    # Load configuration
    config: dict = utils.load_config()

    # Download and unzip news dataset
    path_to_news_dataset: str = config["data"]["news"]["path"]
    download_news_dataset(path_to_news_dataset)
    if args.unzip:
        unzip_and_clean_news_dataset(path_to_news_dataset)

    # Download wikipedia dataset
    download_wikipedia_dataset()


if __name__ == "__main__":
    # Download data -> unzip folders -> clean directory structure (only keep the json files)
    parser = argparse.ArgumentParser()
    parser.add_argument("--unzip", action="store_true", help="Unzip the files after downloading")
    args = parser.parse_args()

    main(args)
