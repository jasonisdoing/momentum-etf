import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from utils.stock_list_io import get_etfs

account_id = "kor_us"
etfs = get_etfs(account_id)

buckets = set()
for etf in etfs:
    b = etf.get("bucket")
    if b is not None:
        buckets.add(b)

print(f"Account: {account_id}")
print(f"Total ETFs: {len(etfs)}")
print(f"Buckets found: {sorted(list(buckets))}")
print(f"Number of buckets: {len(buckets)}")
fbucket_counts = {}
for b in sorted(list(buckets)):
    fbucket_counts[b] = sum(1 for etf in etfs if etf.get("bucket") == b)
print(f"ETFs per bucket: {fbucket_counts}")
