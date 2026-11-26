import time


class Timer():
    def __init__(self, name='time', start=True):
        self.name = name
        if start:
            self.start()

    def start(self):
        self.vals = []
        self.ttin = [time.process_time(), time.perf_counter(), time.time()]

    def measure(self, print_time=True):
        self.vals.append([time.process_time() - self.ttin[0],
                          time.perf_counter() - self.ttin[1],
                          time.time() - self.ttin[2]])
        if print_time:
            print(self)

    def __repr__(self):
        return 'time (%s): %s' % (self.name, str(self.vals))
