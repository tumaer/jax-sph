"""Remove contents of a directory"""

import os
import shutil

if __name__ == "__main__":
    print("Start cleaning stuff")

    for dirname in os.listdir("./"):
        if os.path.isdir(dirname) and ("2D_" in dirname):
            shutil.rmtree(dirname)

    print("Done cleaning!")
