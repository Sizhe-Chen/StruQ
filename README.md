# StruQ: Defending Against Prompt Injection with Structured Queries
[Sizhe Chen](https://sizhe-chen.github.io), [Julien Piet](https://people.eecs.berkeley.edu/~julien.piet), [Chawin Sitawarin](https://chawins.github.io), [David Wagner](https://people.eecs.berkeley.edu/~daw)

[![](https://img.shields.io/badge/USENIX%20Security-2025-e1dd72)](http://arxiv.org/abs/2402.06363) [![](https://img.shields.io/badge/Paper-a8c66c)](http://arxiv.org/pdf/2402.06363)  [![](https://img.shields.io/badge/Website-097770)](https://sizhe-chen.github.io/StruQ-Website) [![](https://img.shields.io/badge/Poster-1b6535)](https://drive.google.com/file/d/1UUz4t43sGqFOPZqNxf8izR--iLAl16QX/view?usp=sharing) [![](https://img.shields.io/badge/Talk-edca82)](https://simons.berkeley.edu/talks/david-wagner-uc-berkeley-2024-10-14) [![](https://img.shields.io/badge/Slides-f47a60)](https://drive.google.com/file/d/1baUbgFMILhPWBeGrm67XXy_H-jO7raRa/view?usp=sharing)

Recent advances in Large Language Models (LLMs) enable exciting LLM-integrated applications, which perform text-based tasks by utilizing their advanced language capabilities. However, as LLMs have improved, so have the attacks against them. Prompt injection attack is listed as the #1 threat to LLM-integrated applications, where an LLM input contains a trusted prompt (instruction) and an untrusted data (user documents, web retrieval, results from API calls, etc) with potentially injected instructions (Ignore previous instructions and …) to arbitrarily manipulate the LLM.

We introduce structured queries, a general approach to tackle this problem. Structured queries separate prompts and data into two channels. We implement a system that supports structured queries. This system is made of (1) a secure front-end that formats a prompt and user data into a special format, and (2) a specially trained LLM that can produce highquality outputs from these inputs. The LLM is trained using a novel fine-tuning strategy: we convert a base (non-instructiontuned) LLM to a structured instruction-tuned model that will only follow instructions in the prompt portion of a query. To do so, we augment standard instruction tuning datasets with examples that also include instructions in the data portion of the query, and fine-tune the model to ignore these. Our system significantly improves resistance to prompt injection attacks, with little or no impact on utility.

# Environment
+ The training requires 4 GPUs, and the testing requires 1 GPU. The code has been tested on 80GB A100s on a slurm cluster. Part of this repo comes from [Alpaca](https://github.com/tatsu-lab/stanford_alpaca).
+ Install environment dependencies
> git clone https://github.com/Sizhe-Chen/StruQ \
> cd StruQ \
> conda create -n struq python==3.10
+ Install package dependencies
> pip install -r requirements.txt
+ Download data dependencies
> python setup.py
+ Configure openai dependencies for utility evaluation: create ```data/openai_configs.yaml``` following ```data/openai_configs_examle.yaml```
+ [optional] Download trained models to play. This command downloads 4 Undefended / StruQ models (llama-7b, Mistral-7B-v0.1).
> python setup.py --model
+ [optional] Automatic and efficient testing by specifying your training/testing slurm configurations in the ```slurm_prefix``` variables in ```run.py```, which generates slurm scripts, run them, and delete them. It supports an additional thread from ```nohup``` to moniter the training, and automatically tests after the training finishes if ```--do_test``` is specified


## Training
+ The ```run.py``` script automatically train multiple models and test them by generating slurm scripts, run them, and delete them.
+ ```nohup python -u run.py -m huggyllama/llama-7b -train SpclSpclSpcl_NaiveCompletion -test none naive ignore completion_real gcg > struq.log 2>&1 &``` stands for training the model with three special delimiters ([MARK] [INST] [COLN]) and Naive+Completion attacks (StruQ-defended model), and test utility and naive, ignore, completion_real, gcg attacks. You may replace NaiveCompletion with None to train an undefended model.
+ Training data size is always 52K, including 26K data that is guaranteed to be unchanged. The data without an input in the remaining 26K samples is also unchanged. Those with an input is prompt-injected by another random sample, with injection method Naive:Completion=1:1


## Testing
+ Running ```run.py``` should trigger the testing (on utility and security) at the end when the model is saved. Logs are saved to the model path.
+ Run only testing by ```python test.py -m huggyllama/llama-7b_SpclSpclSpcl_NaiveCompletion_2024-02-02-00-00-00 -a none naive ignore completion_real gcg```. Log GCG by ```python log.py -m ```