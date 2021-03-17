from glob import glob
import json
from library import P
import numpy as np
from ast import literal_eval as la

p = P("jaro")

LIST_OF_FILE = glob(p.p("report_testing", "*.csv"))

for file in LIST_OF_FILE:
    f = open(file, 'r')
    text = f.read()
    f.close()
    json_ = {}
    report = text.splitlines()
    for thing in report:
        try:
            json_[thing.split(":")[0]] = la(thing.split(":")[1])
        except ValueError:
            json_[thing.split(":")[0]] = np.nan
    with open(file.replace(".csv", ".json"), "w") as f:
        json.dump(json_, f)
