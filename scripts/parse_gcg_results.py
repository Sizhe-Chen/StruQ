from pathlib import Path
import json


MODELS = [
    "llama-7b_SpclSpclSpcl_NaiveCompletion_2024-02-02-00-00-00",
    "llama-7b_TextTextText_None_2024-02-01-00-00-00",
    "mistral-7b_SpclSpclSpcl_NaiveCompletion_2024-02-02-00-00-00",
    "mistral-7b_TextTextText_None_2024-02-01-00-00-00",
]


def main():
    # e.g., logs/llama-7b_SpclSpclSpcl_NaiveCompletion_2024-02-02-00-00-00/gcg/len20_500step_bs512_seed0_l50_t1.0_static_k256
    for model in MODELS:
        dir_path = Path("logs") / model / "gcg/len20_500step_bs512_seed0_l50_t1.0_static_k256"
        num_total, num_success = 0, 0
        missing_ids = set(range(50))
        for p in dir_path.glob("*.jsonl"):
            sample_id = int(p.stem)
            missing_ids.remove(sample_id)
            with p.open("r", encoding="utf-8") as file:
                success = any(not json.loads(line)["passed"] for line in file)
            num_success += int(success)
            num_total += 1
        print(f"{model}: {num_success}/{num_total} ({num_success / num_total:.2%})")
        print("Missing:", list(missing_ids))


if __name__ == "__main__":
    main()
