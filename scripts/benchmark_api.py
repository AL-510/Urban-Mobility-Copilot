"""Benchmark the API endpoint latency and throughput.

Usage:
    python -m scripts.benchmark_api [--url http://localhost:8000] [--requests 50]

Outputs:
    - Latency distribution (mean, p50, p95, p99)
    - Throughput (requests/sec)
    - Error rate
    - Results saved to docs/api_benchmark.json
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Portland downtown test coordinates
TEST_CASES = [
    {
        "name": "Downtown-to-Pearl",
        "origin_lat": 45.5152, "origin_lon": -122.6784,
        "dest_lat": 45.5265, "dest_lon": -122.6836,
    },
    {
        "name": "NW-to-SE",
        "origin_lat": 45.5320, "origin_lon": -122.6900,
        "dest_lat": 45.5050, "dest_lon": -122.6550,
    },
    {
        "name": "Bridge-crossing",
        "origin_lat": 45.5130, "origin_lon": -122.6720,
        "dest_lat": 45.5130, "dest_lon": -122.6600,
    },
]


def run_route_request(base_url: str, test_case: dict, preference: str = "balanced") -> dict:
    """Execute a single route request and return timing info."""
    payload = {
        "origin_lat": test_case["origin_lat"],
        "origin_lon": test_case["origin_lon"],
        "dest_lat": test_case["dest_lat"],
        "dest_lon": test_case["dest_lon"],
        "departure_time": "2024-06-15T08:30:00",
        "preference": preference,
        "max_routes": 3,
        "horizon_minutes": 30,
    }

    t0 = time.time()
    try:
        resp = requests.post(f"{base_url}/api/v1/routes", json=payload, timeout=30)
        elapsed = time.time() - t0
        return {
            "status": resp.status_code,
            "elapsed_ms": round(elapsed * 1000, 1),
            "success": resp.status_code == 200,
            "routes_returned": len(resp.json().get("routes", [])) if resp.status_code == 200 else 0,
            "test_case": test_case["name"],
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "status": 0,
            "elapsed_ms": round(elapsed * 1000, 1),
            "success": False,
            "routes_returned": 0,
            "test_case": test_case["name"],
            "error": str(e),
        }


def benchmark_health(base_url: str, n: int = 10) -> dict:
    """Benchmark health endpoint."""
    times = []
    for _ in range(n):
        t0 = time.time()
        try:
            requests.get(f"{base_url}/health", timeout=5)
            times.append((time.time() - t0) * 1000)
        except Exception:
            pass

    if not times:
        return {"error": "Health endpoint unreachable"}

    return {
        "mean_ms": round(np.mean(times), 1),
        "p50_ms": round(np.median(times), 1),
        "p95_ms": round(np.percentile(times, 95), 1),
    }


def benchmark_network_status(base_url: str, n: int = 10) -> dict:
    """Benchmark network-status endpoint."""
    times = []
    for _ in range(n):
        t0 = time.time()
        try:
            resp = requests.get(f"{base_url}/api/v1/network-status", timeout=5)
            if resp.status_code == 200:
                times.append((time.time() - t0) * 1000)
        except Exception:
            pass

    if not times:
        return {"error": "Network status endpoint unreachable"}

    return {
        "mean_ms": round(np.mean(times), 1),
        "p50_ms": round(np.median(times), 1),
        "p95_ms": round(np.percentile(times, 95), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark API endpoints")
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=30, help="Number of route requests")
    parser.add_argument("--concurrent", type=int, default=1, help="Concurrent requests")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    total_requests = args.requests

    # Check health first
    logger.info(f"Checking API at {base_url}...")
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        health = resp.json()
        logger.info(f"API healthy: {health}")
    except Exception as e:
        logger.error(f"API unreachable: {e}")
        sys.exit(1)

    results = {"api_url": base_url, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # Health endpoint benchmark
    logger.info("Benchmarking /health endpoint...")
    results["health"] = benchmark_health(base_url)
    logger.info(f"  Health: {results['health']}")

    # Network status benchmark
    logger.info("Benchmarking /api/v1/network-status...")
    results["network_status"] = benchmark_network_status(base_url)
    logger.info(f"  Network status: {results['network_status']}")

    # Route endpoint benchmark
    logger.info(f"Benchmarking /api/v1/routes ({total_requests} requests, concurrency={args.concurrent})...")
    preferences = ["balanced", "fastest", "least_risky", "cheapest"]
    all_results = []

    t_start = time.time()

    if args.concurrent > 1:
        with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
            futures = []
            for i in range(total_requests):
                tc = TEST_CASES[i % len(TEST_CASES)]
                pref = preferences[i % len(preferences)]
                futures.append(executor.submit(run_route_request, base_url, tc, pref))

            for f in as_completed(futures):
                all_results.append(f.result())
    else:
        for i in range(total_requests):
            tc = TEST_CASES[i % len(TEST_CASES)]
            pref = preferences[i % len(preferences)]
            result = run_route_request(base_url, tc, pref)
            all_results.append(result)
            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i+1}/{total_requests}")

    total_time = time.time() - t_start

    # Analyze results
    successful = [r for r in all_results if r["success"]]
    failed = [r for r in all_results if not r["success"]]
    latencies = [r["elapsed_ms"] for r in successful]

    route_stats = {
        "total_requests": total_requests,
        "successful": len(successful),
        "failed": len(failed),
        "error_rate": round(len(failed) / total_requests * 100, 1),
        "total_time_s": round(total_time, 1),
        "throughput_rps": round(total_requests / total_time, 2),
    }

    if latencies:
        route_stats.update({
            "latency_mean_ms": round(np.mean(latencies), 1),
            "latency_p50_ms": round(np.median(latencies), 1),
            "latency_p95_ms": round(np.percentile(latencies, 95), 1),
            "latency_p99_ms": round(np.percentile(latencies, 99), 1),
            "latency_min_ms": round(min(latencies), 1),
            "latency_max_ms": round(max(latencies), 1),
        })

    # Per-test-case breakdown
    per_case = {}
    for tc in TEST_CASES:
        case_results = [r for r in successful if r["test_case"] == tc["name"]]
        if case_results:
            case_latencies = [r["elapsed_ms"] for r in case_results]
            per_case[tc["name"]] = {
                "count": len(case_results),
                "mean_ms": round(np.mean(case_latencies), 1),
                "avg_routes": round(np.mean([r["routes_returned"] for r in case_results]), 1),
            }

    route_stats["per_test_case"] = per_case
    results["routes"] = route_stats

    # Print summary
    logger.info("=" * 60)
    logger.info("API BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Total requests: {total_requests}")
    logger.info(f"  Success rate: {len(successful)}/{total_requests} ({100 - route_stats['error_rate']:.0f}%)")
    logger.info(f"  Throughput: {route_stats['throughput_rps']:.2f} req/s")
    if latencies:
        logger.info(f"  Latency (mean): {route_stats['latency_mean_ms']:.1f}ms")
        logger.info(f"  Latency (p50):  {route_stats['latency_p50_ms']:.1f}ms")
        logger.info(f"  Latency (p95):  {route_stats['latency_p95_ms']:.1f}ms")
        logger.info(f"  Latency (p99):  {route_stats['latency_p99_ms']:.1f}ms")
    logger.info(f"\n  Per test case:")
    for name, stats in per_case.items():
        logger.info(f"    {name}: {stats['mean_ms']:.1f}ms avg, {stats['avg_routes']} routes")

    if failed:
        logger.info(f"\n  Failures:")
        for f in failed[:5]:
            logger.info(f"    {f.get('test_case')}: {f.get('error', f'HTTP {f.get(\"status\")}')}")

    # Save results
    output_dir = Path("docs")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "api_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to docs/api_benchmark.json")


if __name__ == "__main__":
    main()
