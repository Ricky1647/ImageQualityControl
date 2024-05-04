import pandas as pd
import json
import sys

# 用來把csv轉成json 方便在reference時候使用 json去作mapping
# csv 透過跟醫生討論獲得對應的label值
def csv2json(filename):
    directory = {}    
    file = pd.read_csv(filename)
    for i in range(len(file)):
        name = file["name"].iloc[i]
        label = file["label"].iloc[i]
        directory.update({f"{name}": f"{label}"})
        with open("qualified.json", "w") as outfile: 
            json.dump(directory, outfile,ensure_ascii=False, indent=4)
    


if __name__ == "__main__":
    filename = sys.argv[1]
    csv2json(filename)
