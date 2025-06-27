import json
import os
import argparse
from datasets import Dataset


def load_dataset(data_path):
    assert os.path.exists(data_path), f"{data_path} does not exist"
    rows = []
    with open(data_path, "r") as file:
        for line in file:
            if line.strip():  # skip empty lines
                rows.append(json.loads(line))
    return rows


def main(args):
    def gen(rows):
        for id, item in enumerate(rows):
            for key, value in item.items():
                item[key] = str(item[key])
            yield item

    splits = ['train', 'valid', 'test']
    for split in splits:
        category_path = os.path.join(args.data_path, f'{split}.jsonl')
        print(f"category_path: {category_path}")
        rows = load_dataset(category_path)
        print(f"Loaded {len(rows)} rows")

        dataset = Dataset.from_generator(gen, gen_kwargs={"rows": rows}, num_proc=args.num_proc)
        dataset.save_to_disk(os.path.join(args.out_dir, f"{split}"))

        print(dataset[:5])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_proc", type=int, default=1)
    args = parser.parse_args()
    main(args)

# python build_dataset.py --data_path /home/yzhang/research/pro2rna/data/output --out_dir /home/yzhang/research/pro2rna/data/build/