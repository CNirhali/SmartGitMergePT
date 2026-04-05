import ipaddress
import timeit
from guardrails import InputValidator

def benchmark_ip_validation():
    print("⚡ Bolt: Benchmarking IP Validation Optimization ⚡")
    validator = InputValidator()

    # Test cases for different IP types
    ips = [
        ipaddress.ip_address('127.0.0.1'),
        ipaddress.ip_address('1.1.1.1'),
        ipaddress.ip_address('::1'),
        ipaddress.ip_address('2002:c0a8:0101::'), # 6to4
        ipaddress.ip_address('2001:0000:4136:e378:8000:63bf:3fff:fdd2'), # Teredo
    ]

    iterations = 50000

    # Measure _is_internal_ip
    print(f"\nRunning _is_internal_ip for {len(ips)} IPs x {iterations} iterations...")
    def run_internal():
        for ip in ips:
            validator._is_internal_ip(ip)

    duration = timeit.timeit(run_internal, number=iterations)
    print(f"Total time: {duration:.4f}s")
    print(f"Average time per call: {(duration / (iterations * len(ips))) * 1e6:.4f} us")

    # Measure _is_loopback_ip
    print(f"\nRunning _is_loopback_ip for {len(ips)} IPs x {iterations} iterations...")
    def run_loopback():
        for ip in ips:
            validator._is_loopback_ip(ip)

    duration = timeit.timeit(run_loopback, number=iterations)
    print(f"Total time: {duration:.4f}s")
    print(f"Average time per call: {(duration / (iterations * len(ips))) * 1e6:.4f} us")

if __name__ == "__main__":
    benchmark_ip_validation()
