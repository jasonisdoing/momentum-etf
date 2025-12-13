import unicodedata

from utils.report import render_table_eaw

headers = ["Category", "Next"]
rows = [["🦾 로봇", "|"], ["🔐 사이버보안", "|"], ["💊 헬스케어", "|"]]
aligns = ["left", "left"]

print("Rendering table...")
lines = render_table_eaw(headers, rows, aligns)
for line in lines:
    print(line)

print("\nDebug Widths:")
from utils.report import normalize


def get_width(s):
    # Copy logic from _disp_width_eaw in utils/report.py manually or verify via import if possible
    # We can inspect what report.py essentially usually does
    from unicodedata import east_asian_width

    s = normalize("NFKC", str(s))
    w = 0
    for ch in s:
        if "\u2500" <= ch <= "\u257f":
            w += 2
            continue
        if unicodedata.category(ch) in ("Mn", "Me", "Cf"):
            print(f"Skipping {ch!r} cat={unicodedata.category(ch)}")
            continue
        eaw = east_asian_width(ch)
        cw = 2 if eaw in ("W", "F", "A") else 1
        print(f"'{ch}' U+{ord(ch):04X} EAW={eaw} W={cw}")
        w += cw
    return w


for r in rows:
    cat = r[0]
    print(f"String: {cat}")
    w = get_width(cat)
    print(f"Total Width: {w}\n")
