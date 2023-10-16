import json
import time
from collections import OrderedDict
from multiprocessing import Lock, Process
import sys
from threading import Thread

import numpy as np
import torch
import math

from scipy.optimize import curve_fit
from ft.COCI_Manager import COCIManager


class COCIIterator:
    def __init__(self,
                 model_name,
                 dataloader,
                 ft_lambda,
                 ck_mode,
                 ts,
                 theta_1,
                 theta_2,
                 epoch,
                 ft_strategy='COCI',
                 fit_interval=None,
                 profile_threshold=None,
                 **ck_kwargs):
        # self._dataloader = dataloader
        self.dataloader = dataloader
        self.epoch = epoch
        self.latest_update_t = 0
        self.latest_snapshot_t = 0
        self.free_mem = 0
        self.ck_size = 0
        self.iter_index = 0
        self.fit_interval = fit_interval
        self.meta_state = {}
        self.iter_list = [0]  # a iter complete time
        self.loss_list = [-1]
        self.iters_online_list = []
        self.loss_online_list = []

        self.start_t = time.time()
        self.ft_lambda = ft_lambda
        self.ck_list = []  # a ck start time
        self.ck_online_list = []
        self.ck_online_index = 0
        self.pre_iter_t = 0

        self.para_profile_flag = False
        self.profile_threshold = profile_threshold

        self.parameters_online = {'a': 0, 'b': 0}
        self.threshold = {'a': 0.1, 'b': 0.1}
        self.snapshot_label = None
        self.async_container = None

        self.COCIManager = COCIManager(model_name, **ck_kwargs)

        self.ft_strategy = ft_strategy
        self.ck_mode = ck_mode
        self.ts_list = []
        if self.ck_mode == 'MANUAL':
            self.parameters = {'a': theta_1, 'b': theta_2}
            self.ts = ts
            if self.ft_strategy == 'COCI':
                self._COCI_strategy(sys_t=0)
            elif self.ft_strategy == 'CCM':
                self._CCM_strategy()
        elif self.ck_mode == 'AUTO':
            self.parameters = {'a': 0, 'b': 0}

        self.test_ck_num = 0
        self.ck_loss = []
        self.ck_forecast_loss = []
        self.ck_cost = 0
        self.ck_flag = 0
        self.ck_cost_index = 0
        self.ck_start_t = []
        self.ck_stop_t = []

    def _general_profile(self):
        def get_ck_size(manager):
            def _get_all_size(ele, sz=0):
                if torch.is_tensor(ele):
                    sz += ele.nelement() * ele.element_size()
                elif isinstance(ele, dict):
                    for k, v in ele.items():
                        sz = _get_all_size(v, sz)
                elif isinstance(ele, list):
                    for v in ele:
                        sz = _get_all_size(v, sz)
                else:
                    sz += sys.getsizeof(ele)

                return sz

            snap_ptr = {}
            size = 0
            for name, ref in manager.ck.destination.items():
                snap_ptr[name] = ref.state_dict()
                size += _get_all_size(snap_ptr[name])
            return size / 1024 / 1024

        dev = max(0, torch.cuda.current_device())
        self.free_mem = (torch.cuda.get_device_properties(dev).total_memory - torch.cuda.max_memory_allocated(
            dev)) / 1024 / 1024
        self.ck_size = get_ck_size(self.COCIManager)

        if self.ck_size < self.free_mem:
            self.snapshot_label = 'GPU'
        else:
            self.snapshot_label = 'CPU'
        self.snapshot_label = 'CPU'

        lock = Lock()
        start_t_process = time.time()
        if self.snapshot_label == 'GPU':
            process = Process(target=self.COCIManager.ck._snapshot_GPU, args=(lock, self.meta_state))
        else:
            process = Process(target=self.COCIManager.ck._snapshot_CPU, args=(lock, self.meta_state))
        process.start()
        process.join()
        terminate_t_process = time.time()
        snapshot_t_process = terminate_t_process - start_t_process

        start_t_thread = time.time()
        if self.snapshot_label == 'GPU':
            thread = Thread(target=self.COCIManager.ck._snapshot_GPU, args=(lock, self.meta_state))
        else:
            thread = Thread(target=self.COCIManager.ck._snapshot_CPU, args=(lock, self.meta_state))
        thread.start()
        thread.join()
        terminate_t_thread = time.time()
        snapshot_t_thread = terminate_t_thread - start_t_thread

        if snapshot_t_thread < snapshot_t_process:
            self.async_container = 'Thread'
        else:
            self.async_container = 'Process'
        self.async_container = 'Thread'

    def _parameter_profile(self):
        def loss(t, a, b):
            return np.exp(a * t + b)

        def fit(x, y):
            x_n = np.array(x)
            y_n = np.array(y)

            # 非线性最小二乘法拟合
            popt, pcov = curve_fit(f=loss, xdata=x_n, ydata=y_n, maxfev=10000)
            # 获取popt里面是拟合系数
            a = popt[0]
            b = popt[1]

            p = {'a': np.float(a), 'b': np.float(b)}
            # print('parameter_a type is {}'.format(type(p['a'])))
            return p

        if self.iter_index == self.profile_threshold:
            current_t = time.time()
            sys_t = (current_t - self.start_t) / 60
            # print('iter_list len is {} and loss_list len is {}'.format(len(self.iter_list), len(self.loss_list)))

            self.parameters = fit(self.iter_list, self.loss_list)
            print('a:{} and b:{}'.format(self.parameters['a'], self.parameters['b']))
            self._COCI_strategy(sys_t)
            self.para_profile_flag = True

    def COCI_save_dynamic(self):
        current_t = time.time()
        sys_t = (current_t - self.start_t) / 60

        # take a checkpoint to make a profile
        if self.iter_index == 10 and len(self.ts_list) == 0:
            self._general_profile()
            self.COCIManager.save(snapshot_label=self.snapshot_label,
                                 async_container=self.async_container,
                                 meta_state=self._get_meta_state())

        if self.iter_index > 0:
            self.pre_iter_t = self.iter_list[-1]

        self.iter_index += 1
        self.iter_list.append(sys_t)
        print('iter:{}, loss:{}'.format(self.iter_list[self.iter_index], self.loss_list[self.iter_index]))

        self._parameter_profile()
        if self.para_profile_flag is True:
            if self.iter_index % self.fit_interval == 0:
                flag = self._online_fitting(sys_t)  # update parameters

                if flag == True:
                    # clear online list
                    print('loss curve parameter is changed')
                    self.ck_online_index = 0
                    self.iters_online_list.clear()
                    self.loss_online_list.clear()
            else:
                self.iters_online_list.append(sys_t)
                self.loss_online_list.append(self.loss_list[self.iter_index - 1])

            # print('length of ck_online_list is {}'.format(len(self.ck_online_list)))
            # print('ck_online_index is {}'.format(self.ck_online_index))
            # print('pre_iter_t is {}'.format(self.pre_iter_t))
            # print('sys_t is {}'.format(sys_t))
            t = time.time()
            # 调整检查点，不能在两个iter之间做
            if self.pre_iter_t < self.ck_online_list[self.ck_online_index] and sys_t >= self.ck_online_list[self.ck_online_index]:
                self.COCIManager.save(self.snapshot_label, self.async_container, self._get_meta_state())
                self.test_ck_num += 1
                print()
                print('ck_num:{}'.format(self.test_ck_num))
                # print('snap_label is {} and async_container is {}'.format(self.snapshot_label, self.async_container))
                self.ck_list.append(sys_t)
                self.ck_online_index += 1
            print('COCIManager.save takes {}'.format(time.time() - t))
        else:
            self.iters_online_list.append(sys_t)
            self.loss_online_list.append(self.loss_list[self.iter_index - 1])  # maybe is wrong

    def COCI_save_statistic(self):
        current_t = time.time()
        sys_t = (current_t - self.start_t) / 60  # take 'min' as unit

        # take a checkpoint to make a profile
        if self.iter_index == 0:
            self._general_profile()

        if self.iter_index > 0:
            self.pre_iter_t = self.iter_list[-1]

        self.iter_index += 1
        self.iter_list.append(sys_t)
        # print('iter:{}, loss:{}'.format(self.iter_list[self.iter_index], self.loss_list[self.iter_index]))

        if len(self.ck_online_list) == 0:
            print('COCI strategy does not come into effect.')
        else:
            t = time.time()
            # 调整检查点，不能在两个iter之间做
            # print('pre_iter_t is {} and sys_t is {}'.format(self.pre_iter_t, sys_t))
            if self.ck_online_index < len(self.ck_online_list):
                if self.pre_iter_t < self.ck_online_list[self.ck_online_index] and sys_t >= self.ck_online_list[
                    self.ck_online_index]:
                    self.COCIManager.save(self.snapshot_label, self.async_container, self._get_meta_state())
                    # print('ck epoch is {}'.format(self.meta_state['epoch']))
                    self.test_ck_num += 1
                    print('ck_num:{}'.format(self.test_ck_num))
                    # print('snap_label is {} and async_container is {}'.format(self.snapshot_label, self.async_container))
                    self.ck_list.append(sys_t)
                    self.ck_online_index += 1

            # print('COCIManager.save takes {}s'.format(time.time() - t))

    def _online_fitting(self, sys_t):
        def loss(t, a, b):
            return np.exp(a * t + b)

        def fit(x, y):
            x_n = np.array(x)
            y_n = np.array(y)

            # 非线性最小二乘法拟合
            popt, pcov = curve_fit(loss, x_n, y_n, maxfev=10000)
            # 获取popt里面是拟合系数
            # print(popt)
            a = popt[0]
            b = popt[1]

            # yvals = func(x_n, a, b)  # 拟合y值
            p = {'a': np.float(a), 'b': np.float(b)}
            # print('parameter_a type is {}'.format(type(p['a'])))
            return p

        self.parameters_online = fit(self.iters_online_list, self.loss_online_list)
        a_online = self.parameters_online['a']
        b_online = self.parameters_online['b']
        a = self.parameters['a']
        b = self.parameters['b']
        if abs(a_online - a) > self.threshold['a']:
            flag = True
        elif abs(b_online - b) > self.threshold['b']:
            flag = True
        else:
            flag = False

        if flag == True:
            self.parameters['a'] = self.parameters_online['a']
            self.parameters['b'] = self.parameters_online['b']
            self._COCI_strategy(sys_t)

        return flag

    def _COCI_strategy(self, sys_t):
        def loss(t, a, b):
            return math.exp(a * t + b)
        def inverse_loss(p, a, b):
            return (math.log(p, math.e) - b)/a

        ts = self._get_ts()
        if self.ck_mode == 'AUTO':
            if len(self.ts_list) == 1 and len(self.ck_list) == 0:
                self.ts_list.clear()

        ck_cost = len(self.ck_list)*ts
        a = self.parameters['a']
        b = self.parameters['b']

        pc = (1 - math.exp(a*ts)) * (1/(2*self.ft_lambda) - 1/(2*a))
        running_t = sys_t - ck_cost
        current_fit_loss = loss(t=running_t, a=a, b=b)
        # current_loss = self.loss_list[-1]
        ck_num = math.floor(current_fit_loss/pc)

        for i in range(ck_num):
            ck_slot = inverse_loss(p=current_fit_loss - pc, a=a, b=b) + ck_cost
            self.ck_online_list.insert(i, ck_slot)
            current_fit_loss -= pc

        print('ck list:')
        # print('ts size is {}'.format(len(self.ts_list)))
        # print('ck_list size is {}'.format(len(self.ck_list)))
        for i in range(len(self.ck_online_list)):
            print('\t {}th ck take in {}min'.format(i+1, self.ck_online_list[i]))

    def _CCM_strategy(self):
        ts = self._get_ts()
        ck_interval = math.sqrt(2 * ts / self.ft_lambda)
        ck_num = round(30.0 / ck_interval)
        ck_slot = 0

        for i in range(ck_num):
            ck_slot += ck_interval
            self.ck_online_list.insert(i, ck_slot)

        print('ck list:')
        for i in range(len(self.ck_online_list)):
            print('\t {}th ck take in {} min'.format(i+1, self.ck_online_list[i]))

    def optimizer_step(self, loss, model, optimizer):
        self.get_loss(loss.item())
        self.weight_update()

        if hasattr(model, 'state_dict'):
            self.COCIManager.ck.destination['model'] = model
        else:
            self.COCIManager.ck.logger.info("No state_dict() method exposed in object{}".format('model'))

        if hasattr(optimizer, 'state_dict'):
            self.COCIManager.ck.destination['optimizer'] = optimizer
        else:
            self.COCIManager.ck.logger.info("No state_dict() method exposed in object{}".format('optimizer'))

        self.COCI_save_statistic()
        # self.COCI_save_dynamic()

        # Determine whether it is the last iteration
        if self.iter_index % len(self.dataloader) == 0:
            epoch = self.iter_index // len(self.dataloader)
            iter_in_epoch = len(self.dataloader)
        else:
            epoch = self.iter_index // len(self.dataloader) + 1
            iter_in_epoch = self.iter_index % len(self.dataloader)
        # print('total iteration number is {}'.format(self.iter_index))
        # print('epoch is {} \t total epoch is {} \t iter in epoch is {}'.format(epoch, self.epoch, iter_in_epoch))
        if epoch == self.epoch and (self.iter_index % len(self.dataloader)) == (len(self.dataloader) - 1):
            self._get_meta_state()
            self.meta_state['is_last_one'] = True

            if len(self.ts_list) != 0:
                sum = 0
                for ts in self.ts_list:
                    sum += ts
                self.meta_state['ts'] = sum / len(self.ts_list)
            else:
                self.meta_state['ts'] = -1

            p = self._fit(self.iter_list, self.loss_list)
            self.meta_state['theta_1'] = p['a']
            self.meta_state['theta_2'] = p['b']

            print()
            print('the last ck')
            self.COCIManager.save(self.snapshot_label, self.async_container, self.meta_state)
            print('theta_1:{} | theta_2:{} | ts:{}'.format(self.meta_state['theta_1'], self.meta_state['theta_2'], self.meta_state['ts']))

    def _fit(self, x, y):
        def loss(t, a, b):
            return np.exp(a * t + b)

        x_n = np.array(x)
        y_n = np.array(y)

        # 非线性最小二乘法拟合
        popt, pcov = curve_fit(loss, x_n, y_n, maxfev=10000)
        # 获取popt里面是拟合系数
        # print(popt)
        a = popt[0]
        b = popt[1]

        # yvals = func(x_n, a, b)  # 拟合y值
        p = {'a': np.float(a), 'b': np.float(b)}
        # print('parameter_a type is {}'.format(type(p['a'])))
        return p

    def recovery(self,
                 gpu=0):
        last_metadata_path, flag = self.COCIManager.recovery(gpu=gpu)
        if flag is True:
            with open(last_metadata_path, 'r+', encoding='ISO-8859-1') as metadata_f:
                js = metadata_f.read()
                metadata_dict = json.loads(js)
            # print(metadata_dict)
            self.parameters['a'] = metadata_dict['theta_1']
            self.parameters['b'] = metadata_dict['theta_2']
            self.ts = metadata_dict['ts']
            ck_loss = metadata_dict['ck_loss']
            # print('ck loss is {}'.format(ck_loss))

            # print(metadata_dict['is_last_one'])
            if metadata_dict['is_last_one'] is False:
                epoch = metadata_dict['epoch']
                print('recovery epoch is {}'.format(epoch))
                iter_in_epoch = metadata_dict['iter_in_epoch']
                self.iter_index = iter_in_epoch + (epoch - 1) * len(self.dataloader)
                if self.ft_strategy == 'COCI':
                    last_start_t = metadata_dict['start_time']
                    ck_t = metadata_dict['ck_time']
                    recovery_t = time.time()
                    rollback_t = recovery_t - ck_t
                    self.start_t = last_start_t + rollback_t
                elif self.ft_strategy == 'CCM':
                    pass

                #  Moves the ck to be created after the error backward by rollback_t
                # if self.ft_strategy == 'COCI':
                #     current_ck_num = len(self.ts_list)
                #     for i in range(current_ck_num, len(self.ck_online_list)):
                #         self.ck_online_list[i] += rollback_t

                # return None, None
                return epoch, iter_in_epoch
            else:
                return self.parameters['a'], self.parameters['b']
        #         return epoch, self.iter_index
        #     return 0, 0
        #     return ck_loss
        # return -1, -1

    def weight_update(self):
        if 'optimizer' in self.COCIManager.ck.destination:
            optimizer = self.COCIManager.ck.destination['optimizer']
            # print('weight update:{}'.format(time.time()))
            # print('ck state in weight update is {}'.format(self.COCIManager.ck.ck_state))

            if self.COCIManager.ck.ck_state == 'snapshot':
                snapshot_start_t = time.time()
                self.ck_start_t.append((snapshot_start_t - self.start_t) / 60)
                while self.COCIManager.ck.ck_state == 'snapshot':
                    self.COCIManager.logger.info("checkpointing!!!")
                    continue
                snapshot_stop_t = time.time()
                self.ck_stop_t.append((snapshot_stop_t - self.start_t) / 60)
                # self.ck_flag = time.time()
                # self.ck_cost = snapshot_stop_t - snapshot_start_t
                ck_cost = (snapshot_stop_t - snapshot_start_t) / 60
                print('ck_cost is {}s'.format(ck_cost * 60))
                self.ts_list.append(ck_cost)
                # self.ck_loss.append(self.loss_list[-1])
                # print('ck start loss is {}'.format(self.loss_list[-1]))

            t = time.time()
            optimizer.step()
            # if self.ck_cost > 0:
            #     if time.time() - self.ck_flag >= self.ck_cost:
            #         self.ck_forecast_loss.append(self.loss_list[-1])
            #         print('ck stop loss should be {}'.format(self.loss_list[-1]))
            #         print('ck_loss length is {} and ck_forecast_loss length is {}'.format(len(self.ck_loss), len(self.ck_forecast_loss)))
            #         self.ck_cost = 0
            # print('optimizer step takes {}s'.format(time.time() - t))
        else:
            self.COCIManager.logger.info("NO Optimizer found")

    def _get_meta_state(self):
        epoch = self.iter_index // len(self.dataloader) + 1
        iter_in_epoch = self.iter_index % len(self.dataloader)
        self.meta_state['epoch'] = epoch
        self.meta_state['iter_in_epoch'] = iter_in_epoch
        self.meta_state['theta_1'] = self.parameters['a']
        self.meta_state['theta_2'] = self.parameters['b']
        self.meta_state['ck_time'] = time.time()
        self.meta_state['ck_iter_index'] = self.iter_index
        self.meta_state['ck_loss'] = self.loss_list[self.iter_index]
        self.meta_state['start_time'] = self.start_t
        # self.meta_state['ck_online_list'] = self.ck_online_list

        self.meta_state['is_last_one'] = False

        if self.ck_mode == 'MANUAL':
            self.meta_state['ts'] = self.ts
        elif self.ck_mode == 'AUTO':
            sum = 0
            for ts in self.ts_list:
                sum += ts
            self.meta_state['ts'] = sum / len(self.ts_list)

        return self.meta_state

    def _get_ts(self):
        if self.ck_mode == 'AUTO':
            sum = 0
            for i in self.ts_list:
                sum += i
            return sum / len(self.ts_list)
        else:
            return self.ts

    def get_loss(self, loss):
        self.loss_list.append(loss)

