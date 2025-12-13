from unicodedata import east_asian_width

emojis = ["🔐", "🦾", "🤖", "🔒"]
for char in emojis:
    print(f"Char: {char}, Width: {east_asian_width(char)}")
