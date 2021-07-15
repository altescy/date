import argparse
import base64
import math
import random
from typing import List


def generate_mask(length: int, mask_ratio: float) -> List[int]:
    mask = [int(random.uniform(0, 1) < mask_ratio) for _ in range(length)]
    return mask


def encode_mask(mask: List[int]) -> str:
    length = len(mask)
    mask_str = "".join(str(x) for x in mask)
    b = int(mask_str, base=2).to_bytes(math.ceil(length / 8), "big")
    return f"{length}.{base64.b64encode(b).decode()}"


def decode_mask(encoded_mask: str) -> List[int]:
    length_str, mask_pattern = encoded_mask.split(".")
    length = int(length_str)
    b = base64.b64decode(mask_pattern.encode())
    mask_value = bin(int.from_bytes(b, "big")).lstrip("0b")
    mask_str = f"{mask_value:>0{length}s}"
    return [int(x) for x in mask_str]


def _run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=str, default=None)
    parser.add_argument("-l", "--length", type=int, default=498)
    parser.add_argument("-n", "--num-masks", type=int, default=25)
    parser.add_argument("-r", "--mask-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--decode", action="store_true")
    args = parser.parse_args()

    if args.decode:
        if args.input is None:
            return
        mask = decode_mask(args.input)
        print("".join(str(x) for x in mask))
        return

    if args.seed is not None:
        random.seed(args.seed)

    for _ in range(args.num_masks):
        mask = generate_mask(args.length, args.mask_ratio)
        encoded_mask = encode_mask(mask)
        print(encoded_mask)


if __name__ == "__main__":
    _run()
