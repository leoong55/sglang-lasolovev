"""One in-flight HiCache readiness snapshot on a dedicated CPU group."""
import torch


class ReadyConsensus:
    def __init__(self, launch):
        self.launch = launch
        self.pending = None

    def submit(self, writes, loads, digest):
        if self.pending is not None:
            raise RuntimeError("HiCache consensus already in flight")
        tensor = torch.tensor([writes, loads, digest, -digest], dtype=torch.int64, device="cpu")
        self.pending = (self.launch(tensor), tensor, digest)

    def finish(self):
        if self.pending is None:
            return 0, 0
        work, tensor, digest = self.pending
        work.wait()  # All ranks finish the previous round, irrespective of local readiness.
        counts = tensor.tolist()
        if counts[-2] != digest or counts[-1] != -digest:
            raise RuntimeError("HiCache snapshot duplicate-reclaim digest diverged across ranks")
        self.pending = None
        return int(counts[0]), int(counts[1])
