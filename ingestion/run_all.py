"""Run several ingestion pipelines in sequence.

  python -m ingestion.run_all                       # all four domains
  python -m ingestion.run_all finance biomedical    # just these
"""
import argparse
import importlib

DOMAINS = ["biomedical", "finance", "customer_support", "general_knowledge"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("domains", nargs="*", help=f"any of: {', '.join(DOMAINS)} (default: all)")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()
    unknown = [d for d in args.domains if d not in DOMAINS]
    if unknown:
        parser.error(f"unknown domain(s): {', '.join(unknown)}")
    for domain in args.domains or DOMAINS:
        print(f"\n=== {domain} ===")
        importlib.import_module(f"ingestion.ingest_{domain}").main(args.max_records)
