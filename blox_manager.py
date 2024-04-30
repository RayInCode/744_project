import jobs
import cluster
import flags
import log
import util
import scheduler
import placement

FLAGS = flags.FLAGS

class Blox_manager():
    def __init__(self, jobs_state: jobs._TFJobs, cluster_state: cluster._Cluster, LOG: log._Log,
                 scheduling_policy: scheduler.Scheduler, placement_policy: placement.JobPlacement):
        self.jobs_state = jobs_state
        self.cluster_state = cluster_state
        self.log = LOG
        self.time = 0
        self.scheduling_policy = scheduling_policy
        self.placement_policy = placement_policy
        self.packings = dict()


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

    def add_new_jobs(self, event: dict):
        if 'start_jobs' in event:
            for s_job in event['start_jobs']:
                #add into runnable list with pending status
                self.jobs_state.move_to_runnable(s_job)

                s_job['remaining_time'] = s_job['duration']
                s_job['remaining_gputime'] = float(s_job['remaining_time'] * s_job['num_gpu'])
                s_job['total_executed_time'] = 0.0
                s_job['total_executed_gputime'] = 0.0
                s_job['calc_executed_time'] = 0.0
                util.print_fn('---- job[%d] is added' % s_job['job_idx'])
    
    def update_runnalble_jobs(self, event: dict):
        #update the status for all running jobs, update metrics for last round, and update values for the comming round
        for rjob in self.jobs_state.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                tmp_oh = rjob['overhead']
                tmp = max(self.time - rjob['last_check_time']-tmp_oh, 0)   
                rjob['remaining_iteration'] -= tmp/rjob['iteration_time_cur']
                rjob['calc_executed_time'] = float(rjob['calc_executed_time'] + tmp/rjob['iteration_time_cur']*rjob['iteration_time'])
                rjob['total_executed_time'] = float(rjob['total_executed_time'] + self.time - rjob['last_check_time'])
                rjob['last_check_time'] = self.time
                rjob['remaining_time'] = rjob['remaining_iteration'] * rjob['iteration_time']
                if FLAGS.gputime:
                    rjob['remaining_gputime'] = float(rjob['remaining_time'] * rjob['num_gpu'])
                    if not FLAGS.know_duration:
                        rjob['total_executed_gputime'] = float(rjob['total_executed_time'] * rjob['num_gpu'])
                # print(self.time, 'check: running ', rjob['job_idx'], rjob['num_gpu'], rjob['total_executed_time'], rjob['calc_executed_time'], rjob['remaining_time'], rjob['duration'], rjob['pending_time'], rjob['iteration_time_cur'], rjob['iteration_time'])
            elif 'PENDING' == rjob['status']:
                tmp = float(self.time - rjob['last_check_time'])
                rjob['pending_time'] = float(rjob['pending_time'] + tmp)
                rjob['last_check_time'] = self.time
                # print(self.time, 'check: pending ', rjob['job_idx'], rjob['num_gpu'], rjob['total_executed_time'], rjob['calc_executed_time'], rjob['remaining_time'], rjob['duration'], rjob['pending_time'], rjob['iteration_time_cur'], rjob['iteration_time'])
            elif 'END' == rjob['status']: #almost impossible
                self.jobs_state.runnable_jobs.remove(rjob)
                print(self.time, 'check: end ', rjob['job_idx'], rjob['total_executed_time'], rjob['duration'])
                pass
            if rjob['status'] != 'END':
                if FLAGS.know_duration: 
                    if FLAGS.gputime:
                        rjob['sort_val']=rjob['remaining_gputime']
                    else:
                        rjob['sort_val']=rjob['remaining_time']
                else:
                    if FLAGS.gputime:
                        rjob['sort_val']=rjob['total_executed_gputime']
                    else:
                        rjob['sort_val']=rjob['total_executed_time']

    def schedule(self) -> dict:
        self.packings = self.scheduling_policy.schedule(self.jobs_state, self.cluster_state)
        return self.packings