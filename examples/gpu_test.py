"""Check that a job gets a GPU and the project env can see it through torch.

Copy it into your project and run `kaya submit gpu_test.py --time 00:05:00`.
"""

# kaya: target=gpu
# kaya: job-name=gpu-test

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("torch is not installed in this env")
        return 1
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    if not torch.cuda.is_available():
        print("no GPU visible")
        return 1
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"gpu {i}: {props.name}, {props.total_memory / 2**30:.1f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
