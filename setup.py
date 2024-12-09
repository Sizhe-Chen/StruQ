import os
import gdown
import argparse
parser = argparse.ArgumentParser(prog='Generating Advprompter Data')
parser.add_argument('--model', default=False, action='store_true')
args = parser.parse_args()


# Download data dependencies
data_urls = [
    'https://huggingface.co/datasets/hamishivi/alpaca-farm-davinci-003-2048-token/resolve/main/davinci_003_outputs.json',
    'https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/refs/heads/main/alpaca_data_cleaned.json',
    'https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json',
    'https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/refs/heads/main/client_configs/openai_configs_example.yaml',
]

os.makedirs('data', exist_ok=True)
for data_url in data_urls:
    data_path = 'data/' + data_url.split('/')[-1]
    if os.path.exists(data_path): print(data_path, 'already exists.'); continue
    cmd = 'wget -P data {data_url}'.format(data_url=data_url, data_path=data_path)
    print(cmd)
    os.system(cmd)


# Download model dependencies
if not args.model: exit()
urls = [
        "https://drive.google.com/drive/folders/1eeJ0MjK6KndRq_tJa1vOpHd85w_nUdyu",
        "https://drive.google.com/drive/folders/1eyV5-UMri8BH6uADhN7OPLKWYzND5Z_-",
        "https://drive.google.com/drive/folders/1D2IRW-0FNwQVqYhYJ-9bRiD3xrIrhaS9",
        "https://drive.google.com/drive/folders/1XSetm-g4lmY6XMretDxL8rp9XMyR8yNb",
    ]
for url in urls:
    gdown.download_folder(url)