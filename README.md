# StruQ: Defending Against Prompt Injection with Structured Queries

## Decription

+ The code is the official implementation of [StruQ: Defending Against Prompt Injection with Structured Queries](https://arxiv.org/abs/2402.06363).
+ The training requires 4 GPUs, and the testing requires 1 GPU. The code has been tested on 80GB A100s on a slurm cluster.
+ Part of this repo comes from [Alpaca](https://github.com/tatsu-lab/stanford_alpaca).
+ This repo is under MIT License.

## Environment
### Packages
+ clone this repo and ```cd SecAlign```
+ create the conda env by running ```conda create -n secalign python==3.10```. If you use another env name, specify it in the ```-e``` in ```run.py```
+ install dependencies by running ```pip install -r requirements.txt```
### Data
+ data/[alpaca_data_clean.json](https://github.com/gururise/AlpacaDataCleaned/blob/main/alpaca_data_cleaned.json): training set
+ data/[alpaca_data.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json): reference set for training-time completion attacks
+ data/[davinci_003_outputs.json](https://huggingface.co/datasets/hamishivi/alpaca-farm-davinci-003-2048-token/resolve/main/davinci_003_outputs.json): testing set for utility and security
### Base LLMs
+ models/[llama-7b](https://huggingface.co/huggyllama/llama-7b)
+ models/[Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
### Undefended SFT LLMs (optional)
+ models/[llama-7b_TextTextText_None_2024-02-01-00-00-00](https://drive.google.com/drive/folders/1eeJ0MjK6KndRq_tJa1vOpHd85w_nUdyu?usp=sharing)
+ models/[mistral-7b_TextTextText_None_2024-02-01-00-00-00](https://drive.google.com/drive/folders/1D2IRW-0FNwQVqYhYJ-9bRiD3xrIrhaS9?usp=sharing)
+ All models can be downloaded at once by installing `gdown` and then run the following commands:
```bash
pip install gdown
mkdir models
python download_models.py
mv llama-7b* mistral* models/
```

## Training

+ The ```run.py``` script automatically train multiple models and test them by generating slurm scripts, run them, and delete them.
+ ```nohup python -u run.py -m llama-7b -train TextTextText_None SpclSpclSpcl_NaiveCompletion -test none naive ignore completion_real > run.log 2>&1 &``` stands for training the first model with three text delimiters (### instruction:) and None attack (undefended model), training the second model with three special delimiters ([MARK] [INST] [COLN]) and Naive+Completion attacks (StruQ-defended model), and test the two models on naive, ignore, completion_real attacks.
+ The whole training data size is always 52K, including 26K clean data. Each attack samples are uniformly and sampled.
+ Only data with input would be injected by the specified attack from another random training sample. Data without input is kept unchanged.

## Testing

+ Running ```run.py``` should trigger the testing (on utility and security) at the end when the model is saved.
+ Run only testing by ```python test.py -m models/llama-7b_SpclSpclSpcl_NaiveCompletion_2024-02-02-00-00-00 -a none naive ignore completion_real```,
+ All training and testing logs are saved to, e.g., ```logs/llama-7b_SpclSpclSpcl_NaiveCompletion_2024-02-02-00-00-00```.
+ Note that the attack success rate (asr) numbers from the code are higher than the actual asr, which should be calculated after manually removing false positive samples.
