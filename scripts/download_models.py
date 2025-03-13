import gdown


def main():
    urls = [
        "https://drive.google.com/drive/folders/1eeJ0MjK6KndRq_tJa1vOpHd85w_nUdyu",
        "https://drive.google.com/drive/folders/1eyV5-UMri8BH6uADhN7OPLKWYzND5Z_-",
        "https://drive.google.com/drive/folders/1D2IRW-0FNwQVqYhYJ-9bRiD3xrIrhaS9",
        "https://drive.google.com/drive/folders/1XSetm-g4lmY6XMretDxL8rp9XMyR8yNb",
    ]
    for url in urls:
        gdown.download_folder(url)


if __name__ == "__main__":
    main()
