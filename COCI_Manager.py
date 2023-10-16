import logging
import os
import re
import shutil
import threading
import time
from collections import OrderedDict
from multiprocessing import Manager, Lock
from os.path import isfile

from ft.COCI_Checkpoint import COCICheckpoint


class COCIManager:
    def __init__(self,
                 model_name: str = 'model',
                 **ck_kwargs):
        self.logger = logging.getLogger(__name__)
        # self.logger.setLevel(logging.INFO)
        if not os.path.exists('./log/COCI_Manager.log'):
            os.makedirs('./log/COCI_Manager.log')
        # logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
        #                     level=logging.DEBUG,
        #                     filename='./log/COCI_Manager.log',
        #                     filemode='a')
        self.pre_ckfile_list = Manager().list()
        self.ck_global_id = 0
        self.ck_dir = './checkpoint'
        self.ck_prefix = 'ck_' + model_name + '_'

        self.ck = COCICheckpoint(**ck_kwargs)
        self.ck_process = None
        self.lock = Lock()
        self.ckname_dict = OrderedDict()
        self.ck_dir_num = 2

        self._initialize_ckdir()

    def save(self,
            snapshot_label:str='CPU',
            async_container:str='Thread',
            meta_state:dict=None):
        # s = time.time()
        self.logger.info("[{}] ENTER SAVE FN".format(time.time()))
        self.ck_global_id += 1
        ck_fname = self.ck_prefix + str(self.ck_global_id)
        ck_filepath = os.path.join(self.ck_dir, ck_fname + '.chk')
        meta_filepath = os.path.join(self.ck_dir, ck_fname + '.meta')

        if self.ck_process is not None:
            if self.ck_process.is_alive():
                if self.ck.ck_state == 'snapshot':
                    self.logger.info('The snapshot of latest checkpoint is taking.')
                    self.ck_process.join()
                elif self.ck.ck_state == 'persist':
                    self.logger.info('The persist of latest checkpoint is taking.')
                    self.ck_process.join()

        if async_container == 'Thread':
            fn = getattr(threading, 'Thread')
        elif async_container == 'Process':
            fn = globals()["Process"]
        else:
            self.logger.info('async_container should be Thread or Process.')
            raise RuntimeError('async_container should be Thread or Process.')

        if meta_state is not None:
            if snapshot_label == 'GPU':
                self.ck_process = fn(target=self.ck._pipeline_ck_GPU,
                                     args=[self.lock, meta_state, ck_filepath, meta_filepath, self.ckname_dict])
            elif snapshot_label == 'CPU':
                self.ck_process = fn(target=self.ck._pipeline_ck_CPU,
                                     args=[self.lock, meta_state, ck_filepath, meta_filepath, self.ckname_dict])
            else:
                raise ValueError('snapshot_label should be CPU or GPU.')
        else:
            self.logger.info('meta_state should not be None.')
            raise ValueError('meta_state should not be None.')

        # print('ck is asynchronized with training')
        self.ck_process.start()

    def recovery(self,
                 gpu=0):
        ck_time = []
        ck_dir_list = os.listdir(self.ck_dir)
        if os.path.exists(self.ck_dir) and len(ck_dir_list) != 0:
            for f in ck_dir_list:
                if isfile(os.path.join(self.ck_dir, f)):
                    if 'chk' in os.path.splitext(f)[1]:
                        ck_time.append((os.path.splitext(f)[0], os.path.getctime(self.ck_dir + '/' + f)))

            # reverse bubble sort
            if len(ck_time) != 0:
                for i in range(1, len(ck_time)):
                    for j in range(0, len(ck_time) - i):
                        if ck_time[j][1] > ck_time[j + 1][1]:
                            ck_time[j], ck_time[j + 1] = ck_time[j + 1], ck_time[j]

                last_ck_name = ck_time.pop()[0]
                last_ck_path = os.path.join(self.ck_dir, last_ck_name + '.chk')
                self.logger.info("Latest checkpoint is {}".format(last_ck_path))

                flag = self.ck._recovery(filepath=last_ck_path, gpu=gpu)
                last_metadata_path = os.path.join(self.ck_dir, last_ck_name + '.meta')
                return last_metadata_path, flag
            else:
                print('There is not ck file!')
                return '', False

    def _initialize_ckdir(self):
        # def atoi(text):
        #     return int(text) if text.isdigit() else text
        #
        # def natural_keys(text):
        #     return [atoi(c) for c in re.split(r'(\d+)', text)]

        if os.path.exists(self.ck_dir):
            ck_time = []
            ck_files = []
            if len(os.listdir(self.ck_dir)) == 0:
                return 0
            for f in os.listdir(self.ck_dir):
                if isfile(os.path.join(self.ck_dir, f)):
                    if 'chk' in os.path.splitext(f)[1]:
                        ck_time.append((os.path.splitext(f)[0], os.path.getctime(self.ck_dir + '/' + f)))

            # reverse bubble sort
            for i in range(1, len(ck_time)):
                for j in range(0, len(ck_time) - i):
                    if ck_time[j][1] > ck_time[j + 1][1]:
                        ck_time[j], ck_time[j + 1] = ck_time[j + 1], ck_time[j]

            if len(ck_time) > self.ck_dir_num:
                for i in range(len(ck_time) - self.ck_dir_num):
                    os.remove(self.ck_dir + '/' + ck_time[i][0] + '.chk')
                    if os.path.exists(self.ck_dir + '/' + ck_time[i][0] + '.meta'):
                        os.remove(self.ck_dir + '/' + ck_time[i][0] + '.meta')

            for i in range(self.ck_dir_num):
                ck_files.insert(0, ck_time.pop()[0])

            self.logger.info(ck_files)
            # ck_files.sort(key=natural_keys)

            for files in ck_files:
                self.pre_ckfile_list.append(files)
                self.ckname_dict[files] = True
                print(files, type(files))
            del ck_files
        else:
            os.makedirs(self.ck_dir)
