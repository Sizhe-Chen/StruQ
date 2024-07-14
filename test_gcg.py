import argparse
import csv
import dataclasses
import logging
import os
import torch

import fastchat
from ml_collections import config_dict

from config import (
    DELIMITERS,
    PROMPT_FORMAT,
    SYS_INPUT,
    TEST_INJECTED_PROMPT,
    TEST_INJECTED_WORD,
)
from gcg.gcg import GCGAttack
from gcg.log import setup_logger
from gcg.utils import Message, Role, SuffixManager, get_nonascii_toks
from struq import jload
from test import load_model_and_tokenizer, test_model_output, form_llm_input, test_parser, load_lora_model

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class CustomConversation(fastchat.conversation.Conversation):
    def get_prompt(self) -> str:
        system_prompt = self.system_template.format(system_message=self.system_message)
        seps = [self.sep, self.sep2]
        ret = system_prompt + self.sep
        for i, (role, message) in enumerate(self.messages):
            if message:
                ret += role + "\n" + message + seps[i % 2]
            else:
                ret += role + "\n"
        return ret
    
    def copy(self):
        return CustomConversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

def gcg(d_item, attack, cfg, data_delm):
    cfg.sample_id = d_item["id"]
    attack._setup_log_file(cfg)
    prompt_no_sys = (
        f"{d_item['instruction']}\n\n{data_delm}\n{d_item['input']}"
        f" {TEST_INJECTED_PROMPT.capitalize()}"
    )
    adv_suffix = attack.run(
        [Message(Role.SYSTEM, SYS_INPUT), Message(Role.USER, prompt_no_sys)], 
        TEST_INJECTED_WORD.lower()
    ).best_suffix
    d_item["input"] += ' ' + TEST_INJECTED_PROMPT.capitalize() + ' ' + adv_suffix
    torch.cuda.empty_cache()
    return d_item

def test_gcg(args):
    setup_logger(True)
    model, tokenizer, frontend_delimiters, training_attacks = load_lora_model(args.model_name_or_path, args.device)

    cfg = config_dict.ConfigDict()
    cfg.name = "gcg"  # Attack name
    cfg.seed = 0  # Random seed
    cfg.log_freq = 20
    cfg.adv_suffix_init = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    # Init suffix length (auto-generated from adv_suffix_init)
    cfg.init_suffix_len = -1
    cfg.num_steps = 500
    cfg.fixed_params = True  # Used fixed scenario params in each iteration
    cfg.allow_non_ascii = False
    cfg.batch_size = 512  # Number of candidates to evaluate in each step
    # NOTE: Reduce mini_batch_size if OOM
    cfg.mini_batch_size = 64 #256  # -1 for full batch (config.batch_size)
    cfg.seq_len = 50  # Max sequence length for computing loss
    cfg.loss_temperature = 1.0  # Temperature for computing loss
    cfg.max_queries = -1  # Max number of queries (default: -1 for no limit)
    cfg.skip_mode = "none"  # "none", "visited", "seen"
    cfg.add_space = False  # Add metaspace in front of target
    cfg.topk = 256
    cfg.num_coords = (1, 1)  # Number of coordinates to change in one step
    cfg.mu = 0.0  # Momentum parameter
    cfg.custom_name = ""
    log_dir = args.model_name_or_path.replace('models', 'logs')
    cfg.log_dir = log_dir
    cfg.sample_id = -1 # to be initialized in every run of the sample

    prompt_template = PROMPT_FORMAT[frontend_delimiters]["prompt_input"]
    inst_delm = DELIMITERS[frontend_delimiters][0]
    data_delm = DELIMITERS[frontend_delimiters][1]
    resp_delm = DELIMITERS[frontend_delimiters][2]

    fastchat.conversation.register_conv_template(
        CustomConversation(
            name="struq",
            system_message=SYS_INPUT,
            roles=(inst_delm, resp_delm),
            sep="\n\n",
            sep2="</s>",
        )
    )

    def eval_func(adv_suffix, messages):
        inst, data = messages[1].content.split(f'\n\n{data_delm}\n')
        return test_model_output([
            prompt_template.format_map({
                "instruction": inst,
                "input": data + ' ' + adv_suffix
            })
        ], model, tokenizer)

    attack = GCGAttack(
        config=cfg,
        model=model,
        tokenizer=tokenizer,
        eval_func=eval_func,
        suffix_manager=SuffixManager(
            tokenizer=tokenizer,
            use_system_instructions=False,
            conv_template=fastchat.conversation.get_conv_template("struq"),
        ),
        not_allowed_tokens=None if cfg.allow_non_ascii else get_nonascii_toks(tokenizer),
    )

    data = [d for d in jload(args.data_path) if d["input"] != ""]
    sample_ids = list(range(len(data))) if args.sample_ids is None else args.sample_ids
    data = [data[i] for i in sample_ids]
    logger.info(f"Running GCG attack on {len(data)} samples {sample_ids}")
    llm_input = form_llm_input(
        data,
        lambda x: gcg(x, attack, cfg, data_delm),
        PROMPT_FORMAT[frontend_delimiters],
        apply_defensive_filter=not (frontend_delimiters == "TextTextText" and training_attacks == "None"),
        defense=args.defense,
        sample_ids=sample_ids,
    )
    in_response, begin_with, outputs = test_model_output(llm_input, model, tokenizer)

    for inpt, outpt in zip(llm_input, outputs):
        logger.info("Final input: %s", inpt)
        logger.info("Final output: %s", outpt[0])

    os.makedirs(log_dir, exist_ok=True)
    print(
        f"\nGCG success rate {in_response} / {begin_with} (in-response / begin_with) on {args.model_name_or_path}, delimiters {frontend_delimiters}, "
        f"training-attacks {training_attacks}, zero-shot defense {args.defense}\n"
    )
    with open(log_dir + "/gcg-" + args.defense + ".csv", "w", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerows([[llm_input[i], s[0], s[1]] for i, s in enumerate(outputs)])

    summary_path = log_dir + "/summary.tsv"
    if not os.path.exists(summary_path):
        with open(summary_path, "w") as outfile: outfile.write("attack\tin-response\tbegin-with\tdefense\n")
    with open(summary_path, "a") as outfile: outfile.write(f"gcg{sample_ids}\t{in_response}\t{begin_with}\t{args.defense}\n")


if __name__ == "__main__":
    test_gcg(test_parser())
