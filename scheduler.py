from jobs import _TFJobs
from cluster import _Cluster
from matching import Blossom_Same, _Packing

class Muri():
    def __init__(self) -> None:
        self.GPU_num_job = dict() # group all runnable jobs by gpu_num
        self.GPU_chose_job = dict() # the indexes of jobs in each group which have been pre-placement
        self.GPU_nums = dict() # index guard for each group
        packings = dict() # dict of _Packing

    def schedule(self, job_state: _TFJobs, cluster_state: _Cluster):
        '''clear GPU_num_job, GPU_chosen_job, GPU_nums'''
        self.GPU_num_job.clear()
        self.GPU_chose_job.clear()
        self.GPU_nums.clear()

        '''reset all runnable jobs to unpacked'''
        self.group_jobs_by_gpu_nums(self, job_state)
        self.block_job_by_priority(self, job_state, cluster_state)

        '''run packing algorithm'''
        packings = Blossom_Same.run(self.GPU_num_job, cluster_state.num_gpu)
        pass


    '''recalculate GPU_num_job and GPU_choes_job and GPU_num'''
    def group_jobs_by_gpu_nums(self, job_state: _TFJobs):
        '''sort jobs by priority'''

    ''''''
    def block_job_by_priority(self, job_state:_TFJobs, cluster_state: _Cluster):
        '''clear packing_used for each jobs'''
        '''clear cluster states'''


