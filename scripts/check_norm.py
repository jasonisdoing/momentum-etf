from unicodedata import east_asian_width, normalize

s = "🔐"
norm = normalize("NFKC", s)
print(f"Original: {s} (len {len(s)})")
print(f"Normalized: {norm} (len {len(norm)})")
print(f"EAW Orig: {east_asian_width(s)}")
print(f"EAW Norm: {east_asian_width(norm)}")
