from time import time


class Timer:
    def __init__(self):
        self.t = time()
        self.tk = False

    def tick(self):
        self.t = time()
        self.tk = True

    def toc(self, tick_again=False):
        if not self.tk:
            raise RuntimeError('not ticked yet!')
        self.tk = False
        before_t = self.t
        cur_t = time()
        if tick_again:
            self.t = cur_t
            self.tk = True
        return cur_t - before_t
