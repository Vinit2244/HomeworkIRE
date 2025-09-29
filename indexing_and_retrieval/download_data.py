# ======================== IMPORTS ========================
import os
import utils
import subprocess

# ======================= FUNCTIONS =======================
def download_news_dataset(destination: str) -> None:
    print("Downloading news dataset...")

    # Make sure the destination directory exists
    os.makedirs(destination, exist_ok=True)

    # Clone the repository
    repo_url = "https://github.com/Webhose/free-news-datasets.git"
    subprocess.run(["git", "clone", repo_url, destination], check=True)

    print(f"News dataset downloaded at {destination}\n")


def download_wikipedia_dataset() -> None:
    print("Use wikipedia dataset directly using datasets library.\n")


# ======================== MAIN ==========================
def main():
    # Load configuration
    config: dict = utils.load_config()

    # Download datasets
    download_news_dataset(config["data"]["news"]["path"])
    download_wikipedia_dataset()


if __name__ == "__main__":
    main()
