import numpy as np
from types import SimpleNamespace


def args():
    def TR_callback(args):
        print("TR_callback called")
        print(args.hoge)

    hoge = "hoge"
    return locals()


ar = args()
ar = SimpleNamespace(**ar)
ar.TR_callback(ar)
