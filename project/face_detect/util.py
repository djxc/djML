import time
import numpy as np

def calculate_bbox(xy):
    """获取人脸的矩形范围， 输入的xy为中心点以及长短半轴"""
    a = 1.0
    x1=int(float(xy[3])) - a*int(float(xy[1]))
    y1=int(float(xy[4])) - a*int(float(xy[0]))
    x2=int(float(xy[3])) + a*int(float(xy[1]))
    y2=int(float(xy[4])) + a*int(float(xy[0]))
    return [x1, y1, x2, y2]

def process_bar(percent, loss, acc, epoch, start_str='', end_str='', total_length=0):
    bar = ''.join(["%s" % '='] * int(percent * total_length)) + ''
    if loss is not None:
        bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent * 100) + end_str + " epoch " + str(epoch) + " loss " + str(loss)
    else:
        bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent*100) + end_str + " epoch " + str(epoch)
    print(bar, end='', flush=True)

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()