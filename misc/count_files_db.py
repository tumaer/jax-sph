"""How to count number of files in a direcory"""

import os

if __name__ == "__main__":
    src = "datasets/2D_DB_5740_10kevery100"
    dirs = os.listdir(src)
    dirs.sort()
    for dir in dirs:
        files = os.listdir(os.path.join(src, dir))
        fail_message = " failed!" if len(files) != 402 else ""
        print(f"In {dir} there are {len(files)} files. {fail_message}")

    print(f"Total number of dirs: {len(dirs)}")
