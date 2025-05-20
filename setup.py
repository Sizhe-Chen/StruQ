import os
#import gdown
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
model_paths = [
    'huggyllama/llama-7b_SpclSpclSpcl_None_2025-03-12-01-01-20',
    'mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2025-03-12-01-02-08',
    'huggyllama/llama-7b_SpclSpclSpcl_NaiveCompletion_2025-03-12-01-02-37',
    'mistralai/Mistral-7B-v0.1_SpclSpclSpcl_NaiveCompletion_2025-03-15-03-25-16',
]

for model_path in model_paths:
    if os.path.exists(model_path): print(model_path, 'already exists.'); continue
    model_dir = model_path.split('/')[0]
    os.makedirs(model_dir, exist_ok=True)
    cmd = 'wget -P {model_dir} https://dl.fbaipublicfiles.com/SecAlign/{model_path} && unzip {model_path} -d {model_dir} && rm {model_path}'.format(model_path=model_path + '.zip', model_dir=model_dir)
    print(cmd)
    os.system(cmd)
