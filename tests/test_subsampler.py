# Usage:
# python -m tests.test_subsampler
from datasets import load_from_disk

from src.util import Subsampler

if __name__ == "__main__":
    path = "./data/dummy_data_tokenized_deid_text_postal_code_borough"
    data = load_from_disk(path)["train"]
    print(f"loaded {data}, length is {len(data)}")
    subsampler = Subsampler(42, data)
    res = subsampler.subsample(10)
    print(f"after subsample: {res}")
