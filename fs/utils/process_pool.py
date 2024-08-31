from time import sleep 
#
from tqdm import tqdm
class MyProcessPool():
    def __init__(self, size = 4) -> None:
        self.size = size
        assert size > 0
        self._processes = []
        self._close = False
        self.idx = 0
        self._running = 0

    def append(self, process):
        if self._close:
            return
        self._processes.append(process)

    def _start(self):
        if self.idx >= len(self._processes):
            return
        p = self._processes[self.idx]
        self.idx += 1
        self._running += 1
        p.start()
        return p

    def start(self):
        waitting = []
        # bar = tqdm(total=len(self._processes), desc="Waiting for all process done")
        while self.idx < len(self._processes):
            for i in range(self._running, self.size):
                if self.idx >= len(self._processes):
                    continue
                p = self._start()
                waitting.append(p)
            
            while len(waitting) == self.size:
                alives = []
                for w in waitting:
                    w.join(1)
                    if w.is_alive():
                        alives.append(w)
                waitting = alives
                self._running = len(waitting)
            sleep(1)
            # bar.update()
            # print(self._running)

    def close(self):
        self._close = True
        
    def join(self):
        for p in tqdm(self._processes):
            p.join()
