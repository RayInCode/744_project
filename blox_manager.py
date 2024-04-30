import jobs
import cluster
import flags
import log

FLAGS = flags.FLAGS

class Blox_manager():
    def __init__(self, job_state: jobs._TFJobs, cluster_state: cluster._Cluster, LOG: log._Log):
        self.jobs_state = job_state
        self.cluster_state = cluster_state
        self.log = LOG
        self.time = 0


    def predict_completed_jobs(self) -> list:
        return self.jobs_state.predict_completed_jobs()
    
    def pop_arriving_jobs(self) -> list:
        return self.jobs_state.pop_arriving_jobs(self.time)
    
    def handle_completed_jobs(self, event: dict):
        if 'end_jobs' in event:
            for e_job in event['end_jobs']:
                tmp = float(event['time'] - e_job['last_check_time']) 
                e_job['total_executed_time'] = float(e_job['total_executed_time'] + tmp)
                #job completes
                self.cluster_state.release_job_res(e_job)
                # CLUSTER.release_gpus(e_job)
                self.log.job_complete(e_job, event['time'])
                self.jobs_state.runnable_jobs.remove(e_job)
                # print("11111111111", e_job['job_idx'], e_job['num_gpu'], e_job['duration'], e_job['end_time']-e_job['start_time'])

    
    def update_runnalble_jobs(self, event: dict):
        pass

