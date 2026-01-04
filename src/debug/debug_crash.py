import torch
import ray
import sys

def test_cuda():
    print("--- Testing CUDA ---")
    print(f"PyTorch Version: {torch.__version__}")
    try:
        if torch.cuda.is_available():
            print(f"CUDA Available: Yes")
            print(f"Device Count: {torch.cuda.device_count()}")
            print(f"Device Name: {torch.cuda.get_device_name(0)}")
            x = torch.tensor([1.0]).cuda()
            print(f"Tensor on GPU: {x}")
        else:
            print("CUDA Available: NO")
    except Exception as e:
        print(f"CRASH during CUDA check: {e}")
        # Crash might not be caught if it's a segfault/abort, but good to try.

@ray.remote(num_gpus=1)
def ray_task(x):
    import torch
    print(f"Worker: CUDA Available? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Worker: Device {torch.cuda.get_device_name(0)}")
        t = torch.tensor([x]).cuda()
        return t.item() * x
    return x * x

def test_ray():
    print("\n--- Testing Ray ---")
    if not ray.is_initialized():
        ray.init()
    
    try:
        ref = ray_task.remote(10)
        res = ray.get(ref)
        print(f"Ray Task Result (10*10): {res}")
    except Exception as e:
        print(f"Ray Task Failed: {e}")

if __name__ == "__main__":
    test_cuda()
    test_ray()