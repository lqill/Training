from glob import glob
import json
from library import P

p = P("jaro")

LIST_OF_FILE = glob(p.p("hasil", "*.txt"))

for file in LIST_OF_FILE:
    f = open(file, 'r')
    text = f.read()
    f.close()
    json_ = {}
    report = text.splitlines()
    for thing in report:
        json_[thing.split(",")[0]] = thing.split(",")[1]
    with open(file.replace(".txt", ".json"), "w") as f:
        json.dump(json_, f)
