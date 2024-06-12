import sys

if __name__ == "__main__":
    user_input = sys.stdin.read()
    data = user_input.split("\n")
    index = 0
    for i in range(1,len(data)-1):
        users = data[i].split("|")[-1]
        users = users.split(" ")
        if (len(users) == 2):
            print(f"{index}")
            exit(1)
        index+=1
    print("")