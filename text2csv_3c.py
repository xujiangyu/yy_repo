import pandas as pd
import sys

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <text-file> <csv-file>")
    exit(0)

txtfn = sys.argv[1]
csvfn = sys.argv[2]

toks = []
with open(txtfn) as fd:
    for line in fd:
        tok = line.rstrip('\n').split('\t')
        toks.append([None if t == '\\N' else t for t in tok])
df = pd.DataFrame(toks[1:], columns=toks[0])
df.to_csv(csvfn, index=False)
