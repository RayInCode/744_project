from __future__ import print_function
import csv
import re
import sys
import types
import time
import math
#parse args
import argparse
import copy
import os

import numpy as np
import util
import flags
import jobs
import cluster
import log
import cvxpy as cp
from cvxpy import SolverError
from matching import Blossom_Same, _Packing
import scheduler
import placement
import blox_manager

sys.setrecursionlimit(1000000000)
# import hosts
# import placement_scheme as scheme
# import cmd

# profiled overhead of start for each workloads
overhead_dict = {1:dict(), 2:dict(), 4:dict(), 8:dict(), 16:dict(), 32:dict()}
overhead_dict[1] = {'vgg16':7, 'vgg19':7, 'resnet18':4, 'shufflenet_v2_x1_0':4, 'bert':10, 'gpt2':10, 'a2c':38, 'dqn':5}
overhead_dict[2] = {'vgg16':7, 'vgg19':7, 'resnet18':5, 'shufflenet_v2_x1_0':4, 'bert':10, 'gpt2':10, 'a2c':39, 'dqn':5}
overhead_dict[4] = {'vgg16':8, 'vgg19':8, 'resnet18':5, 'shufflenet_v2_x1_0':5, 'bert':10, 'gpt2':10, 'a2c':39, 'dqn':6}
overhead_dict[8] = {'vgg16':9, 'vgg19':9, 'resnet18':7, 'shufflenet_v2_x1_0':5, 'bert':10, 'gpt2':10, 'a2c':46, 'dqn':6}
overhead_dict[16] = {'vgg16':9, 'vgg19':10, 'resnet18':7, 'shufflenet_v2_x1_0':7, 'bert':10, 'gpt2':10, 'a2c':46, 'dqn':8}
overhead_dict[32] = {'vgg16':10, 'vgg19':10, 'resnet18':7, 'shufflenet_v2_x1_0':8, 'bert':10, 'gpt2':10, 'a2c':46, 'dqn':8}
#parse input arguments
flags.DEFINE_string('trace_file', 'tf_job.csv',
                '''Provide TF job trace file (*.csv, *.txt).
                    *.csv file, use \',\' as delimiter; *.txt file, user \' \' as deliminter. 
                    Default file is tf_job.csv ''')
flags.DEFINE_string('log_path', 'result-' + time.strftime("%Y%m%d-%H-%M-%S", time.localtime()),
                '''Simulation output folder, including cluster/node/gpu usage trace, pending job_queue info.
                Default folder is result-[time]''')
flags.DEFINE_string('scheme', 'yarn',
                '''
                Job placement scheme:
                0.count, just resource counting, without assignment (which gpu, which cpu)
                1.yarn, ms yarn
                2.random
                3.crandom (consolidate + random)
                4.greedy
                5.balance
                6.cbalance (consolidate + balance)
                Default is yarn''')
flags.DEFINE_string('schedule', 'fifo',
                '''
                Job schedule scheme:
                1.fifo
                2.shortest, shortest-remaining-time job first
                3.shortest-gpu, shortest-remaining-gputime job first 
                4.dlas, discretized las 
                5.dlas-gpu, dlas using gpu time
                6.antman, AntMan
                7.themis, Themis
                8.multi-resource-blossom-same-gpu(-unaware), match jobs with same #GPU using blossom algorithm using gputime (unaware job duration)
                Default is fifo''')
flags.DEFINE_integer('num_switch', 1, 
                '''Part of cluster spec: the number of switches in this cluster, default is 1''')
flags.DEFINE_integer('num_node_p_switch', 32, 
                '''Part of cluster spec: the number of nodes under a single switch, default is 32''')
flags.DEFINE_integer('num_gpu_p_node', 8, 
                '''Part of cluster spec: the number of gpus on each node, default is 8''')
flags.DEFINE_integer('num_cpu_p_node', 64,
                '''Part of cluster spec: the number of cpus on each node, default is 64''')
flags.DEFINE_integer('mem_p_node', 256,
                '''Part of cluster spec: memory capacity on each node, default is 128''')
flags.DEFINE_string('cluster_spec', None,
                '''Part of cluster spec: cluster infra spec file, 
                this file will overwrite the specs from num_switch, num_node_p_switch, and num_gpu_p_node
                Spec format:
                    num_switch,num_node_p_switch,num_gpu_p_node
                    int,int,int''')
# for multi_resource sharing
flags.DEFINE_integer('multi_resource', 4, 
                '''Part of job spec: the num of resources used for each job, default is 4''')
flags.DEFINE_integer('packing_num', 4,  
                '''maximum number of packing jobs''')
flags.DEFINE_float('weight_lbd', 0.0, '''The factor of the lower bound of expected weight (i jobs packing of n resources: i/n)''')
flags.DEFINE_boolean('autopack', True, '''Unpack job if the combined normalized tput is slower than 1''')
flags.DEFINE_boolean('print', False, 
                '''Enable print out information, default is False''')
flags.DEFINE_boolean('flush_stdout', True, 
                '''Flush stdout, default is True''')
flags.DEFINE_version('0.1')
flags.DEFINE_integer('round_interval', 1440,
                '''Interval for each round of scheduling''')
flags.DEFINE_integer('ordering', 1,
                '''set the ordering for scheduler''')

FLAGS = flags.FLAGS

#prepare JOBS list
job_generator = jobs.JOBS

#get host info
cluster_state = cluster.CLUSTER

#get LOG object
LOG = log.LOG


def parse_job_file(trace_file):
    #check trace_file is *.csv
    fd = open(trace_file, 'r')
    deli = ','
    if ((trace_file.find('.csv') == (len(trace_file) - 4))):
        deli = ','
    elif ((trace_file.find('.txt') == (len(trace_file) - 4))):
        deli = ' '

    reader = csv.DictReader(fd, delimiter = deli) 
    ''' Add job from job trace file'''
    keys = reader.fieldnames
    util.print_fn('--------------------------------- Read TF jobs from: %s ---------------------------------' % trace_file) 
    util.print_fn('    we get the following fields:\n        %s' % keys)
    job_idx = 0
    for row in reader:
        #add job into JOBS
        job_generator.add_job(row)
        # job_generator.read_job_info(job_idx, 'num_gpu')
        job_idx += 1

    assert job_idx == len(job_generator.job_list) 
    assert job_generator.num_job == len(job_generator.job_list) 
    # job_generator.print_all_job_size_info()
    job_generator.sort_all_jobs()
    # print(lp.prepare_job_info(job_generator.job_list[0]))
    util.print_fn('---------------------------------- Get %d TF jobs in total ----------------------------------' % job_idx)
    # job_generator.read_all_jobs()
    fd.close()

def parse_cluster_spec():
    if FLAGS.cluster_spec:
        print(FLAGS.cluster_spec)
        spec_file = FLAGS.cluster_spec
        fd = open(spec_file, 'r')
        deli = ','
        if ((spec_file.find('.csv') == (len(spec_file) - 4))):
            deli = ','
        elif ((spec_file.find('.txt') == (len(spec_file) - 4))):
            deli = ' '
        reader = csv.DictReader(fd, delimiter = deli) 
        keys = reader.fieldnames
        util.print_fn(keys)
        if 'num_switch' not in keys:
            return
        if 'num_node_p_switch' not in keys:
            return
        if 'num_gpu_p_node' not in keys:
            return
        if 'num_cpu_p_node' not in keys:
            return
        if 'mem_p_node' not in keys:
            return
        
        ''' there should be only one line remaining'''
        assert reader.line_num == 1

        ''' get cluster spec '''
        for row in reader:
            # util.print_fn('num_switch %s' % row['num_switch'])
            FLAGS.num_switch = int(row['num_switch'])
            FLAGS.num_node_p_switch = int(row['num_node_p_switch'])
            FLAGS.num_gpu_p_node = int(row['num_gpu_p_node'])
            FLAGS.num_cpu_p_node = int(row['num_cpu_p_node'])
            FLAGS.mem_p_node = int(row['mem_p_node'])
        fd.close()

    util.print_fn("num_switch: %d" % FLAGS.num_switch)
    util.print_fn("num_node_p_switch: %d" % FLAGS.num_node_p_switch)
    util.print_fn("num_gpu_p_node: %d" % FLAGS.num_gpu_p_node)
    util.print_fn("num_cpu_p_node: %d" % FLAGS.num_cpu_p_node)
    util.print_fn("mem_p_node: %d" % FLAGS.mem_p_node)

    '''init infra'''
    cluster_state.init_infra()
    # util.print_fn(lp.prepare_cluster_info())
    util.print_fn('--------------------------------- End of cluster spec ---------------------------------')
    return 


'''
Allocate job resource
'''
def try_get_job_res(job, not_place=False):
    '''
    select placement scheme
    '''
    if 'antman' in FLAGS.schedule:
        ret = cluster_state.antman_placement(job)
    elif FLAGS.scheme == 'yarn':
        ret = cluster_state.ms_yarn_placement(job, not_place)
    elif FLAGS.scheme == 'random':
        ret = cluster_state.random_placement(job)
    elif FLAGS.scheme == 'crandom':
        ret = cluster_state.consolidate_random_placement(job)
    elif FLAGS.scheme == 'greedy':
        ret = cluster_state.greedy_placement(job)
    elif FLAGS.scheme == 'gandiva':
        ret = cluster_state.gandiva_placement(job)
    elif FLAGS.scheme == 'count':
        ret = cluster_state.none_placement(job)
    else:
        ret = cluster_state.ms_yarn_placement(job)
    if ret == True:
        # job['status'] = 'RUNNING'
        pass
    return ret

def cal_shortest_expected_remaining(job_data, a):
    data = job_data['data']
    idx = next(x[0] for x in enumerate(data) if x[1] > a)

    if idx == (job_data['num'] - 1):
        return data[idx]

    num = job_data['num'] - 1 - idx 
    return round(sum(data[idx: (job_data['num'] - 1)]) * 1.0 / num, 2)


def main():
    
    ''' Parse input'''
    parse_job_file(FLAGS.trace_file)  #parse jobs from trace-data and append to job_generator
    parse_cluster_spec() # parse cluster config file into FLAG
    ''' prepare logging '''
    LOG.init_log()

    # lp.placement(job_generator.job_list[0])
    ''' Prepare jobs'''
    job_generator.prepare_job_start_events()

    job_state = jobs.JOBS
    # # init policies
    placement_policy = placement.JobPlacement()
    scheduling_policy = scheduler.Muri()

    blox_mgr = blox_manager.Blox_manager(job_state, cluster_state, LOG)
    round_interval = FLAGS.round_interval
    # while True:
    #     job_state.predict_completed_jobs() # keep a list for predict-complete jobs in job_state
    #     arriving_jobs = job_generator.pop_arriving_jobs(blox_mgr.time)
    #     event_sim['end_jobs']
    #     '''handle all activated jobs in last round''' 
    #     job_state.handle_completed_jobs()  # check out completed jobs in this round, and update their states
    #     job_state.update_runnalble_jobs()  # update the satus of other uncompleted jobs
        
    #     '''handle arriving jobs when last round is running'''
    #     new_jobs = job_generator.generate()
    #     job_generator.add_new_jobs(new_jobs)

    #     blox_mgr.time += round_interval
    '''
    new jobs are added to the end of the ending queue
    but in the queue, shortest (gpu) job first be served
    and pack other jobs with the same #GPU according to 
    graph matching
    '''
    round_interval = FLAGS.round_interval
    event = dict()
    event['time'] = 0
    event['end_jobs'] = []
    event['start_jobs'] = []

    while (len(job_generator.job_events) + len(job_state.runnable_jobs))> 0:
        print("Event Time: ", blox_mgr.time)

        event['time'] = blox_mgr.time
        event['end_jobs'] = []
        event['start_jobs'] = []
        event['end_jobs'] = blox_mgr.predict_completed_jobs()
        event['start_jobs'] =  job_generator.pop_arriving_jobs(blox_mgr.time)
        blox_mgr.handle_completed_jobs(event) 
        # # for ending jobs, release gpu
        # if 'end_jobs' in event:
        #     for e_job in event['end_jobs']:
        #         tmp = float(blox_mgr.time - e_job['last_check_time']) 
        #         e_job['total_executed_time'] = float(e_job['total_executed_time'] + tmp)
        #         #job completes
        #         cluster_state.release_job_res(e_job)
        #         # cluster_state.release_gpus(e_job)
        #         LOG.job_complete(e_job, blox_mgr.time)
        #         job_state.runnable_jobs.remove(e_job)
        #         # print("11111111111", e_job['job_idx'], e_job['num_gpu'], e_job['duration'], e_job['end_time']-e_job['start_time'])

        # blox_mgr.update_runnalble_jobs(event)
        #for new-start jobs, add to runnable
        if 'start_jobs' in event:
            for s_job in event['start_jobs']:
                #add into runnable list with pending status
                job_state.move_to_runnable(s_job)

                s_job['remaining_time'] = s_job['duration']
                s_job['remaining_gputime'] = float(s_job['remaining_time'] * s_job['num_gpu'])
                s_job['total_executed_time'] = 0.0
                s_job['total_executed_gputime'] = 0.0
                s_job['calc_executed_time'] = 0.0
                util.print_fn('---- job[%d] is added' % s_job['job_idx'])

        gputime = True
        know_duration = False
        #update the status for all running jobs, update metrics for last round, and update values for the comming round
        for rjob in job_state.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                tmp_oh = rjob['overhead']
                tmp = max(blox_mgr.time - rjob['last_check_time']-tmp_oh, 0)   
                rjob['remaining_iteration'] -= tmp/rjob['iteration_time_cur']
                rjob['calc_executed_time'] = float(rjob['calc_executed_time'] + tmp/rjob['iteration_time_cur']*rjob['iteration_time'])
                rjob['total_executed_time'] = float(rjob['total_executed_time'] + blox_mgr.time - rjob['last_check_time'])
                rjob['last_check_time'] = blox_mgr.time
                rjob['remaining_time'] = rjob['remaining_iteration'] * rjob['iteration_time']
                if gputime:
                    rjob['remaining_gputime'] = float(rjob['remaining_time'] * rjob['num_gpu'])
                    if not know_duration:
                        rjob['total_executed_gputime'] = float(rjob['total_executed_time'] * rjob['num_gpu'])
                # print(blox_mgr.time, 'check: running ', rjob['job_idx'], rjob['num_gpu'], rjob['total_executed_time'], rjob['calc_executed_time'], rjob['remaining_time'], rjob['duration'], rjob['pending_time'], rjob['iteration_time_cur'], rjob['iteration_time'])
            elif 'PENDING' == rjob['status']:
                tmp = float(blox_mgr.time - rjob['last_check_time'])
                rjob['pending_time'] = float(rjob['pending_time'] + tmp)
                rjob['last_check_time'] = blox_mgr.time
                # print(blox_mgr.time, 'check: pending ', rjob['job_idx'], rjob['num_gpu'], rjob['total_executed_time'], rjob['calc_executed_time'], rjob['remaining_time'], rjob['duration'], rjob['pending_time'], rjob['iteration_time_cur'], rjob['iteration_time'])
            elif 'END' == rjob['status']: #almost impossible
                job_state.runnable_jobs.remove(rjob)
                print(blox_mgr.time, 'check: end ', rjob['job_idx'], rjob['total_executed_time'], rjob['duration'])
                pass
            if rjob['status'] != 'END':
                if know_duration: 
                    if gputime:
                        rjob['sort_val']=rjob['remaining_gputime']
                    else:
                        rjob['sort_val']=rjob['remaining_time']
                else:
                    if gputime:
                        rjob['sort_val']=rjob['total_executed_gputime']
                    else:
                        rjob['sort_val']=rjob['total_executed_time']

        #sort jobs with shortest first
        job_state.runnable_jobs.sort(key = lambda e:e.__getitem__('sort_val'))
        
        #group runnable jobs with according to gpu num, in each group, jobs are in order
        run_jobs = list()
        preempt_jobs = list()
        GPU_num_job = dict()
        GPU_chosen_job = dict()
        GPU_nums = dict()
        required_gpu = 0
        for rjob in job_state.runnable_jobs:
            # assert rjob['packing_used'] < 2
            rjob['packing_used'] = 0    #reset to unpacking status
            num_gpu = rjob['num_gpu']
            if num_gpu not in GPU_num_job:
                GPU_num_job[num_gpu] = list()
            GPU_num_job[num_gpu].append(rjob)
            if num_gpu not in GPU_chosen_job:
                GPU_chosen_job[num_gpu] = 0
                GPU_nums[num_gpu] = 0

        #scan / execute jobs one by one
        cluster_state.empty_infra() # release all resource for comming round

        #select jobs to schedule, select one can go with other three(at most) which require the same gpu num
        for rjob in job_state.runnable_jobs: 
            if rjob['packing_used'] == 1:
                continue
            ret = try_get_job_res(rjob, True) # the key function here
            num_gpu = rjob['num_gpu']
            if ret == True:
                up_bd = min(GPU_chosen_job[num_gpu]+FLAGS.packing_num, len(GPU_num_job[num_gpu]))
                GPU_nums[num_gpu] += 1
                for tmp_id in range(GPU_chosen_job[num_gpu], up_bd):
                    GPU_num_job[num_gpu][tmp_id]['packing_used']=1
                GPU_chosen_job[num_gpu] = up_bd
        
        #truncate the GPU_num_job to keep chosen ones for next scheduling
        for key in GPU_num_job.keys():
            GPU_num_job[key] = GPU_num_job[key][:GPU_chosen_job[key]]
            required_gpu += GPU_chosen_job[key]*key

        # print('before packing')
        # for key,value in GPU_num_job.items():
        #     print(key, 'GPU(s): ', len(value))
        #     print([rjob['job_idx'] for rjob in value])


        # time_before = time.time()
        # matching algorithm
        packings = Blossom_Same.run(GPU_num_job, cluster_state.num_gpu)
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
                            rjob = job_state.find_runnable_job(job.job_idx)
                            new_packing.append((key, _Packing(rjob), rjob['sort_val']))
                    else:
                        sort_val = min([job_state.find_runnable_job(job.job_idx)['sort_val'] for job in packing.packing_jobs])
                        new_packing.append((key, packing, sort_val))
            new_packing.sort(key=lambda e:e[2])
            cluster_state.empty_infra()

            #if we unpack some jobs, we need to check the placement avalibility again
            packings = dict()
            for packing in new_packing:
                rjob = job_state.find_runnable_job(packing[1].packing_jobs[0].job_idx)
                if 'RUNNING'==rjob['status']:
                    if 'placements' in rjob:
                        del rjob['placements'][:]
                ret = try_get_job_res(rjob, True)
                if ret==True:
                    if packing[0] not in packings:
                        packings[packing[0]] = list()
                    packings[packing[0]].append(packing[1])

        # print("_______________________new placement____________________")
        # deal with the packing plan
        cluster_state.empty_infra()

        #final placement let packing with more num_gpu reuqired place first
        job_state.runnable_jobs.sort(key = lambda e:e.__getitem__('num_gpu'), reverse=True)
        tmp_job_placement = dict()
        for rjob in job_state.runnable_jobs:
            # print("after packing: ", rjob['job_idx'], rjob['placements'])
            if 'RUNNING' == rjob['status']:
                if 'placements' in rjob: 
                    del rjob['placements'][:]
            ret = False
            for key, value in packings.items():
                for packing in value:
                    for packing_job in packing.packing_jobs:
                        if packing_job.job_idx == rjob['job_idx']:
                            packing_cur = packing
                            ret = True
                            break
                    if ret:
                        break
                if ret:
                    break
            if ret:
                rjob['iteration_time_cur'] = packing_cur.calc_iteration_time()
                rjob['packing'] = packing_cur
                rjob['overhead'] = 0
                for pjob_ in packing_cur.packing_jobs:
                    pjob = job_state.find_runnable_job(pjob_.job_idx)
                    if pjob['model_name'] in overhead_dict[rjob['num_gpu']]:
                        rjob['overhead'] += overhead_dict[rjob['num_gpu']][pjob['model_name']]
                    else:
                        rjob['overhead'] += 10
                    # rjob['overhead'] = 0
                # print(rjob['job_idx'], [pjob.job_idx for pjob in packing_cur.packing_jobs], rjob['iteration_time'], rjob['iteration_time_cur'])
                # if rjob['iteration_time']/rjob['iteration_time_cur']>len(packing_cur.packing_jobs):
                    # print("111111111", rjob['job_idx'], rjob['iteration_time_cur']/rjob['iteration_time'])
                # print("11111111111111 job: ", rjob['job_idx'], rjob['num_gpu'], rjob['iteration_time_cur'], rjob['iteration_time'], (rjob['iteration_time_cur']-rjob['iteration_time'])/rjob['iteration_time'])
                # print([pjob.job_idx for pjob in packing_cur.best_permutation])
                if rjob['job_idx'] in tmp_job_placement:
                    rjob['placements'] = tmp_job_placement[rjob['job_idx']]
                    # print("other")
                else:
                    ret_1 = try_get_job_res(rjob)
                    if not ret_1:
                        print(f"job {rjob['job_idx']} is unable to place")
                        if rjob['status'] == 'RUNNING':
                            preempt_jobs.append(rjob)
                        continue
                    # assert ret_1==True
                    for packing_job in packing_cur.packing_jobs:
                        tmp_job_placement[packing_job.job_idx] = copy.deepcopy(rjob['placements'])
                    # print('first')
                # print(rjob['placements'])
                if sys.maxsize == rjob['start_time']:
                    rjob['start_time'] = blox_mgr.time
                if rjob['status'] == 'PENDING':
                    run_jobs.append(rjob)
            else:
                if rjob['status'] == 'RUNNING':
                    preempt_jobs.append(rjob)
                continue

        for job in preempt_jobs:
            job['status'] = 'PENDING'
            job['preempt'] = int(job['preempt'] + 1)
            # job['packing_used'] = 0
        # print("-----placement-------")
        for job in run_jobs:
            job['status'] = 'RUNNING'
            job['resume'] = int(job['resume'] + 1)
            # job['packing_used'] = 1
            # print(job['placements'])
            #        

        # get the next end_event
        # del end_events[:]
        # for rjob in job_state.runnable_jobs:
        #     if 'RUNNING' == rjob['status']:
        #         end_time = float(blox_mgr.time + rjob['remaining_iteration']*rjob['iteration_time_cur'])
        #         # print(blox_mgr.time, rjob['job_idx'], rjob['remaining_time'], rjob['iteration_time'], rjob['iteration_time_cur'], end_time)
        #         tmp_dict = util.search_dict_list(end_events, 'time', end_time)
        #         if tmp_dict == None:
        #             #not found, add the time into to job_events
        #             tmp_dict = dict()
        #             tmp_dict['time'] = end_time
        #             tmp_dict['end_jobs'] = list()
        #             tmp_dict['end_jobs'].append(rjob)
        #             end_events.append(tmp_dict)
        #         else:
        #             tmp_dict['end_jobs'].append(rjob)
        # end_events.sort(key = lambda e:e.__getitem__('time'))
        # blox_mgr.time += round_interval

        blox_mgr.time += round_interval



        LOG.checkpoint(blox_mgr.time)     

        

if __name__ == '__main__':
    # print('Hello world %d' % 2)
    main()
