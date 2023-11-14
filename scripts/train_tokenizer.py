
import os
import fire
from pathlib import Path

import sentencepiece as spm
from datasets import load_dataset

dataset = load_dataset("b-mc2/sql-create-context", split='train')
vocab_size = 3072

def main(vocab_size=3072, data_cache_dir="data"):
    # file path for sentencepiece input.
    text_dump = Path(data_cache_dir) / 'text_dump.txt'

    # output file prefix path for sentencepiece
    model_prefix = os.path.join(data_cache_dir, f"tok{vocab_size}")

    if not os.path.exists(text_dump):
        with open(text_dump, "w", encoding="utf-8") as f:
            for ix, sample in enumerate(dataset):
                sample_text = []
                for key in sample.keys():
                    sample_text.append(key + ' : ' + sample[key])
                sample_text = '\n'.join(sample_text)
                f.write(sample_text + '\n\n')

    print(f"text dump size is: {os.path.getsize(text_dump) / 1024 / 1024:.2f} MB")

    spm.SentencePieceTrainer.train(input=text_dump,
                                    model_prefix=model_prefix,
                                    model_type="bpe",
                                    vocab_size=vocab_size,
                                    self_test_sample_size=0,
                                    input_format="text",
                                    character_coverage=1.0,
                                    num_threads=os.cpu_count(),
                                    split_digits=True,
                                    allow_whitespace_only_pieces=True,
                                    byte_fallback=True,
                                    unk_surface=r" \342\201\207 ",
                                    normalization_rule_name="identity")

if __name__ == "__main__":
    fire.Fire(main)
