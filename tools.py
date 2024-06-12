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
    

# 用來把新加入的資料分開 因為有些資料可能有labeled qualified 有些沒有
def isolate(fileTotal, fileB):
    with open(fileTotal) as f:
        file_all = f.read().splitlines()
    with open(fileB) as f:
        file_cur = f.read().splitlines()
    for i in range(len(file_cur)):
        file_all.remove(file_cur[i])
    with open("unlabeled_data.txt", "w") as f:
        for i in file_all:
            f.write(f"{i}\n")


if __name__ == "__main__":
    # filename = sys.argv[1]
    # csv2json(filename)
    isolate(sys.argv[1], sys.argv[2])
