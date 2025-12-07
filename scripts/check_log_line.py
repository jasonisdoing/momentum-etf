def inspect_line(line):
    print(f"Line: {line.strip()}")
    print("Chars:", end=" ")
    for ch in line:
        if ord(ch) > 127:
            print(f"{ch}(U+{ord(ch):04X})", end=" ")
    print()
    from unicodedata import east_asian_width

    width = 0
    for ch in line:
        # Simplified logic from my report.py
        import unicodedata

        if unicodedata.category(ch) in ("Mn", "Me", "Cf"):
            continue
        eaw = east_asian_width(ch)
        if eaw in ("W", "F", "A"):
            width += 2
        else:
            width += 1
    print(f"Calculated Width: {width}")


lines = [
    "|  4 | ARKQ | ARK Autonomous Technology & Robotics ETF                  | 🦾 로봇       | BUY  |      1 |  76.20 |   +0.0% |  114 | $8,719.12 | -$141.10 |   -1.6% | -$141.10 |   -1.6% | 12.5% | 16.9 | 71.5 | 60일 | ✅ 신규 매수                  |",
    "| 12 | BUG  | Global X Cybersecurity ETF                                | 🔐 사이버보안 | WAIT |      0 |  34.09 |   +0.0% |    0 |     $0.00 |    $0.00 |       - |    $0.00 |       - |  0.0% |  7.0 | 64.8 | 22일 |                               |",
]

for line in lines:
    inspect_line(line)
