from unicodedata import east_asian_width

vs16 = "\ufe0f"
print(f"VS16: {repr(vs16)}, EAW: {east_asian_width(vs16)}")
