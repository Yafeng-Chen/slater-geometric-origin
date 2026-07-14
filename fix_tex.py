import re

with open('slater_geometric_origin.tex', 'r') as f:
    content = f.read()

# Replace 4+ backslashes followed by [4pt] with exactly 2 backslashes followed by [4pt]
# In Python raw regex: r'\\\\+' matches 2+ literal backslashes
# In regex replacement: r'\\\\' means 2 literal backslashes
content = re.sub(r'\\\\+\[4pt\]', r'\\\\[4pt]', content)

with open('slater_geometric_origin.tex', 'w') as f:
    f.write(content)

print("Fixed")
# Verify with hex dump
with open('slater_geometric_origin.tex', 'rb') as f:
    data = f.read()
idx = data.find(b'[4pt]')
print(f"First [4pt] at byte {idx}")
print(f"Bytes before: {data[idx-4:idx+10]}")
