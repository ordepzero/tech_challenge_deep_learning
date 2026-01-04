import os
import sys

def check_fips():
    print("Checking FIPS status...")
    try:
        with open("/proc/sys/crypto/fips_enabled", "r") as f:
            status = f.read().strip()
            print(f"/proc/sys/crypto/fips_enabled: {status}")
            if status == "1":
                print("WARNING: FIPS mode is ENABLED. This is likely causing the s2n/PyArrow crash.")
            else:
                print("FIPS mode is disabled (0).")
    except FileNotFoundError:
        print("/proc/sys/crypto/fips_enabled not found.")
    except Exception as e:
        print(f"Error checking FIPS: {e}")

    print("\nPython info:")
    print(sys.version)

if __name__ == "__main__":
    check_fips()