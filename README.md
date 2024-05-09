# Environment
+ clone this repo and ```cd StruQ```
+ download data/[alpaca_data_clean.json](https://github.com/gururise/AlpacaDataCleaned/blob/main/alpaca_data_cleaned.json) as training set, data/[alpaca_data.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) as reference training data for completion attacks, data/[davinci_003_outputs.json](https://huggingface.co/datasets/hamishivi/alpaca-farm-davinci-003-2048-token/resolve/main/davinci_003_outputs.json) as testing set.
+ create the conda env by running ```conda create -n struq python==3.10```. If you use another env name, specify it in the ```-e``` arg of ```run.py```
+ install dependencies by running ```pip install -r requirements.txt```
+ install [alpaca_eval](https://github.com/tatsu-lab/alpaca_eval) for utility evaluation by running ```pip install alpaca-eval```
+ [optional] download [undefended alpaca](https://drive.google.com/drive/folders/1eeJ0MjK6KndRq_tJa1vOpHd85w_nUdyu?usp=sharing) and [StruQ-defended alpaca]() to ```models/```, reproduce the utility evaluation by downloading the pre-saved GPT-4 judgement from AlpacaEval to data/[annotations_seed0_configs.json](https://drive.google.com/file/d/1-TQKGHTEYrIyB6awBv9moAOEHJ0E5E9w/view?usp=sharing).

# Training
+ Run multiple exps, each with multiple attacks, by, e.g., ```python run.py -m llama-7b -train TextTextText_None SpclSpclSpcl_NaiveCompletion -test none naive ignore completion_real```
+ The whole training data size is always 52K, including 26K clean data. Each attack samples are uniformly and sampled.
+ Only data with input would be injected by the specified attack from another random training sample. Data without input is kept unchanged.

# Testing
+ Running ```run.py``` should trigger the testing (on utility and security) at the end when the model is saved. 
+ Run only testing by ```python test.py -m models/llama-7b_SpclSpclSpcl_NaiveCompletion_2024-02-02-00-00-00 -a none naive ignore completion_real```, 
+ All training and testing logs are saved to, e.g., ```logs/llama-7b_SpclSpclSpcl_NaiveCompletion_2024-02-02-00-00-00```.