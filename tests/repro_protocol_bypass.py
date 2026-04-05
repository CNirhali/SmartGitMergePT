import re
from urllib.parse import urlparse, unquote

dangerous_protocol_re = re.compile(
    r'(j\s*+a\s*+v\s*+a\s*+s\s*+c\s*+r\s*+i\s*+p\s*+t|v\s*+b\s*+s\s*+c\s*+r\s*+i\s*+p\s*+t|d\s*+a\s*+t\s*+a|'
    r'f\s*+i\s*+l\s*+e|g\s*+o\s*+p\s*+h\s*+e\s*+r|p\s*+h\s*+p|j\s*+a\s*+r|'
    r'd\s*+i\s*+c\s*+t|l\s*+d\s*+a\s*+p)\s*+:',
    re.IGNORECASE
)

urls = [
    "javascript:alert(1)",
    "j\navascript:alert(1)",
    "j%0aavascript:alert(1)",
    "j%20a%20v%20a%20s%20c%20r%20i%20p%20t:alert(1)",
    "%6a%61%76%61%73%63%72%69%70%74:alert(1)"
]

print(f"{'URL':<40} | {'Regex Match':<12} | {'Unquoted Match':<12}")
print("-" * 70)

for url in urls:
    match = "YES" if dangerous_protocol_re.search(url) else "NO"
    unquoted_url = unquote(url)
    unquoted_match = "YES" if dangerous_protocol_re.search(unquoted_url) else "NO"
    print(f"{url:<40} | {match:<12} | {unquoted_match:<12}")
