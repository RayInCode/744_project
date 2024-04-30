from jobs import _TFJobs
from cluster import _Cluster
from matching import Blossom_Same, _Packing
import flags

FLAGS = flags.FLAGS

class Scheduler(object):
    def __init__(self) -> None:
        pass
    
    def scheduler(self, job_state: _TFJobs, cluster_state: _Cluster):
        raise NotImplementedError("This method should be overridden by subclasses")


class Muri(Scheduler):
    def __init__(self) -> None:
        self.GPU_num_job = dict() # group all runnable jobs by gpu_num
        self.GPU_chosen_job = dict() # the indexes of jobs in each group which have been pre-placement
        self.GPU_nums = dict() # index guard for each group
        self.packings = dict() # dict of _Packing

    def schedule(self, jobs_state: _TFJobs, cluster_state: _Cluster) :
        # clear GPU_num_job, GPU_chosen_job, GPU_nums and packins'''
        self.GPU_num_job.clear()
        self.GPU_chosen_job.clear()
        self.GPU_nums.clear()
        self.packings.clear()
    

        #sort jobs with shortest first
        jobs_state.runnable_jobs.sort(key = lambda e:e.__getitem__('sort_val'))

        for rjob in jobs_state.runnable_jobs:
            # assert rjob['packing_used'] < 2
            rjob['packing_used'] = 0    #reset to unpacking status
            num_gpu = rjob['num_gpu']
            if num_gpu not in self.GPU_num_job:
                self.GPU_num_job[num_gpu] = list()
            self.GPU_num_job[num_gpu].append(rjob)
            if num_gpu not in self.GPU_chosen_job:
                self.GPU_chosen_job[num_gpu] = 0
                self.GPU_nums[num_gpu] = 0

        #scan / execute jobs one by one
        cluster_state.empty_infra() # release all resource for comming round

        #select jobs to schedule, select one can go with other three(at most) which require the same gpu num
        for rjob in jobs_state.runnable_jobs: 
            if rjob['packing_used'] == 1:
                continue
            ret = cluster_state.ms_yarn_placement(rjob, True) # the key function here
            num_gpu = rjob['num_gpu']
            if ret == True:
                up_bd = min(self.GPU_chosen_job[num_gpu]+FLAGS.packing_num, len(self.GPU_num_job[num_gpu]))
                self.GPU_nums[num_gpu] += 1
                for tmp_id in range(self.GPU_chosen_job[num_gpu], up_bd):
                    self.GPU_num_job[num_gpu][tmp_id]['packing_used']=1
                self.GPU_chosen_job[num_gpu] = up_bd
        
        #truncate the GPU_num_job to keep chosen ones for next scheduling
        for key in self.GPU_num_job.keys():
            self.GPU_num_job[key] = self.GPU_num_job[key][:self.GPU_chosen_job[key]]

        # print('before packing')
        # for key,value in GPU_num_job.items():
        #     print(key, 'GPU(s): ', len(value))
        #     print([rjob['job_idx'] for rjob in value])


        # time_before = time.time()
        # matching algorithm
        packings = Blossom_Same.run(self.GPU_num_job, cluster_state.num_gpu)
        # print('after packing', time.time()-time_before)
        # for key, value in packings.items():
        #     for packing in value:
        #         print([packing_job.job_idx for packing_job in packing.packing_jobs], end=':::')
        #         print('gpu', [packing_job.num_gpu for packing_job in packing.packing_jobs])
        
        
        if FLAGS.autopack:
            new_packing = list()
            for key, value in packings.items():
                for packing in value:
                    itertime_all = packing.calc_iteration_time()
                    itertime_sum = sum([job.iteration_time for job in packing.packing_jobs])
                    if itertime_all/itertime_sum >1:
                        print('unpack: ', [job.job_idx for job in packing.packing_jobs])
                        for job in packing.packing_jobs:
                            rjob = jobs_state.find_runnable_job(job.job_idx)
                            new_packing.append((key, _Packing(rjob), rjob['sort_val']))
                    else:
                        sort_val = min([jobs_state.find_runnable_job(job.job_idx)['sort_val'] for job in packing.packing_jobs])
                        new_packing.append((key, packing, sort_val))
            new_packing.sort(key=lambda e:e[2])
            cluster_state.empty_infra()

            #if we unpack some jobs, we need to check the placement avalibility again
            packings = dict()
            for packing in new_packing:
                rjob = jobs_state.find_runnable_job(packing[1].packing_jobs[0].job_idx)
                if 'RUNNING'==rjob['status']:
                    if 'placements' in rjob:
                        del rjob['placements'][:]
                ret = cluster_state.ms_yarn_placement(rjob, True)
                if ret==True:
                    if packing[0] not in packings:
                        packings[packing[0]] = list()
                    packings[packing[0]].append(packing[1])

        return packings



        # '''reset all runnable jobs to unpacked'''
        # self.group_jobs_by_gpu_nums(self, jobs_state)
        # self.block_job_by_priority(self, jobs_state, cluster_state)

        # #truncate the GPU_num_job to keep chosen ones for next scheduling
        # for key in self.GPU_num_job.keys():
        #     self.GPU_num_job[key] = self.GPU_num_job[key][:self.GPU_chosen_job[key]]

        # '''run packing algorithm'''
        # packings = Blossom_Same.run(self.GPU_num_job, cluster_state.num_gpu)
        # pass


    '''recalculate GPU_num_job and GPU_choes_job and GPU_num'''
    def group_jobs_by_gpu_nums(self, jobs_state: _TFJobs):
        # sort all runnable jobs by executed_time in ascendant
        jobs_state.runnable_jobs.sort(key = lambda e:e.__getitem__('sort_val'))

        for rjob in jobs_state.runnable_jobs:
            # assert rjob['packing_used'] < 2
            rjob['packing_used'] = 0    #reset to unpacking status
            num_gpu = rjob['num_gpu']
            if num_gpu not in self.GPU_num_job:
                self.GPU_num_job[num_gpu] = list()
            self.GPU_num_job[num_gpu].append(rjob)
            if num_gpu not in self.GPU_chosen_job:
                self.GPU_chosen_job[num_gpu] = 0
                self.GPU_nums[num_gpu] = 0



    def block_job_by_priority(self, job_state:_TFJobs, cluster_state: _Cluster):
        cluster_state.empty_infra() # release all resource for comming round

        #select jobs to schedule, select one can go with other three(at most) which require the same gpu num
        for rjob in self.jobs_state.runnable_jobs: 
            if rjob['packing_used'] == 1:
                continue
            ret = cluster_state.ms_yarn_placement(rjob, True) # the key function here
            num_gpu = rjob['num_gpu']
            if ret == True:
                up_bd = min(self.GPU_chosen_job[num_gpu]+FLAGS.packing_num, len(self.GPU_num_job[num_gpu]))
                self.GPU_nums[num_gpu] += 1
                for tmp_id in range(self.GPU_chosen_job[num_gpu], up_bd):
                    self.GPU_num_job[num_gpu][tmp_id]['packing_used']=1
                self.GPU_chosen_job[num_gpu] = up_bd

        

