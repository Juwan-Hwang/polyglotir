#!/usr/bin/env python
"""req_id collision test — Phase 1 deliverable.

Simulates 1000+ IR entries with random content, generates 4-digit req_ids,
and measures the collision rate.

Per spec §2 Layer 1:
    "req_id 4 位短哈希，阶段 1 模拟 1000 条测碰撞，>1% 则扩 6~8 位"

The test also runs 6-digit and 8-digit variants for comparison, and outputs
a CSV with the full results.

Usage::

    python scripts/req_id_collision_test.py
    python scripts/req_id_collision_test.py --n 5000
"""

from __future__ import annotations

import csv
import hashlib
import random
import string
import sys
from collections import Counter
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "processed" / "phase1"


# ── req_id generation (mirrors SilpIR.generate_req_id) ───────────────


def generate_req_id(content: str, length: int = 4) -> str:
    """Generate a short hex hash from *content*.

    This is the same algorithm as ``SilpIR.generate_req_id``.
    """
    return hashlib.sha256(content.encode()).hexdigest()[:length]


# ── Test content generators ───────────────────────────────────────────


def random_content(n: int) -> list[str]:
    """Generate *n* unique content strings for req_id testing.

    Uses realistic-looking IR content: verb + entity values + constraints.
    """
    verbs = [
        "!CANCEL", "!START", "!EMAIL", "!FETCH", "!PROCESS",
        "!TRANSLATE", "!SWITCH", "!BOOK", "!ROUTE", "!SEARCH",
        "!UPDATE", "!SUGGEST",
    ]
    targets = [
        "flight", "hotel", "order_42", "ticket", "data", "report",
        "meeting", "payment", "subscription", "invoice", "reservation",
        "delivery", "shipment", "booking", "account", "profile",
    ]
    locations = [
        "Beijing", "Shanghai", "New_York", "London", "Tokyo",
        "Paris", "Berlin", "Sydney", "Toronto", "Singapore",
    ]

    contents: list[str] = []
    seen: set[str] = set()

    for _ in range(n * 3):  # oversample to get n unique
        verb = random.choice(verbs)
        target = random.choice(targets)
        loc = random.choice(locations)
        time = f"t+{random.randint(1, 48)}h"
        # Add random suffix to increase uniqueness
        suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        content = f"{verb}|{target}|{loc}|{time}|{suffix}"

        if content not in seen:
            seen.add(content)
            contents.append(content)

        if len(contents) >= n:
            break

    return contents


# ── Collision test ────────────────────────────────────────────────────


def test_collision(n: int, length: int) -> dict[str, object]:
    """Run a single collision test.

    Args:
        n: Number of unique entries to simulate.
        length: req_id length (4, 6, or 8).

    Returns:
        Dict with test results.
    """
    contents = random_content(n)
    if len(contents) < n:
        print(
            f"  [warn] could only generate {len(contents)} unique contents "
            f"(requested {n})",
            file=sys.stderr,
        )

    ids = [generate_req_id(c, length) for c in contents]

    # Count collisions
    counter = Counter(ids)
    unique_ids = len(counter)
    collision_count = n - unique_ids  # entries that share an ID with another
    collision_rate = collision_count / n * 100 if n else 0

    # Find the actual collision groups (IDs with >1 entry)
    collision_groups = {k: v for k, v in counter.items() if v > 1}

    # Theoretical birthday-problem estimate
    space = 16 ** length  # hex space
    # Probability of at least one collision ≈ 1 - e^(-n²/(2*space))
    import math
    birthday_prob = 1 - math.exp(-(n ** 2) / (2 * space))

    return {
        "n": n,
        "req_id_length": length,
        "unique_ids": unique_ids,
        "collisions": collision_count,
        "collision_rate_pct": round(collision_rate, 4),
        "collision_groups": len(collision_groups),
        "max_group_size": max(collision_groups.values()) if collision_groups else 1,
        "id_space": space,
        "birthday_prob_pct": round(birthday_prob * 100, 4),
        "verdict": "PASS" if collision_rate < 1.0 else "FAIL (>1% threshold)",
    }


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="SILP req_id collision test (Phase 1)"
    )
    parser.add_argument(
        "--n", type=int, default=1000,
        help="Number of entries to simulate (default: 1000)",
    )
    parser.add_argument(
        "--lengths", nargs="*", type=int, default=[4, 6, 8],
        help="req_id lengths to test (default: 4 6 8)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"SILP req_id Collision Test — Phase 1", file=sys.stderr)
    print(f"  Entries:  {args.n}", file=sys.stderr)
    print(f"  Lengths:  {args.lengths}", file=sys.stderr)
    print(f"  Seed:     {args.seed}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    results: list[dict[str, object]] = []
    for length in args.lengths:
        print(f"  Testing length={length}...", end="", file=sys.stderr, flush=True)
        result = test_collision(args.n, length)
        results.append(result)
        print(
            f"  collisions={result['collisions']} "
            f"({result['collision_rate_pct']}%) "
            f"→ {result['verdict']}",
            file=sys.stderr,
        )

    # ── Decision ───────────────────────────────────────────────
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Decision:", file=sys.stderr)

    length4 = next((r for r in results if r["req_id_length"] == 4), None)
    if length4:
        rate = length4["collision_rate_pct"]
        if rate >= 1.0:
            print(
                f"  4-digit req_id collision rate = {rate}% ≥ 1% threshold.",
                file=sys.stderr,
            )
            print(
                f"  → MUST expand to 6-8 digits per spec.",
                file=sys.stderr,
            )
            length6 = next((r for r in results if r["req_id_length"] == 6), None)
            if length6 and length6["collision_rate_pct"] < 1.0:
                print(
                    f"  6-digit rate = {length6['collision_rate_pct']}% < 1% "
                    f"→ recommend 6 digits.",
                    file=sys.stderr,
                )
        else:
            print(
                f"  4-digit req_id collision rate = {rate}% < 1% threshold.",
                file=sys.stderr,
            )
            print(
                f"  → 4-digit req_id is sufficient for {args.n} entries.",
                file=sys.stderr,
            )
    print(f"{'='*60}\n", file=sys.stderr)

    # ── Write CSV ──────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "req_id_collision.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"  Results: {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
