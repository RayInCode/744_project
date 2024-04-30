from jobs import _TFJobs
from cluster import _Cluster

class JobPlacement():
    def __init__(self) -> None:
        pass

    def place(self, jobs_state: _TFJobs, cluster_state: _Cluster):
        pass


class YarnPlacement(JobPlacement):
    def __init__(self) -> None:
        pass

    def place(self, JOBS: _TFJobs, cluster_state: _Cluster, packing):
        pass
