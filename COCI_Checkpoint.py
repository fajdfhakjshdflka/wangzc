import copy
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from collections.abc import Mapping
from sys import getsizeof

import torch

class COCICheckpoint:
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        # self.logger.setLevel(logging.INFO)
        logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                            level=logging.DEBUG,
                            filename='./log/COCI_Checkpoint.log',
                            filemode='a')
        self.latest_snapshot = None
        self.destination = OrderedDict()
        self.ck_complete = None
        self.ck_state = 'idle'

        for name, ref in kwargs.items():
            if hasattr(ref, 'state_dict'):
                self.destination[name] = ref
            else:
                self.logger.info("Skipping object `{}` in CF Checkpointing. No state_dict() method exposed".format(name))

        self.num_tracking = len(self.destination.keys())

        if self.num_tracking == 0:
            raise ValueError("Nothing to track")

    def __getstate__(self):
        d = self.__dict__.copy()
        if 'logger' in d:
            d['logger'] = d['logger'].name
        return d

    def __setstate__(self, d):
        if 'logger' in d:
            d['logger'] = logging.getLogger(d['logger'])
        self.__dict__.update(d)

    def _snapshot_GPU(self,
                      lock,
                      meta_data):
        try:
            if self.ck_state == 'idle':
                with lock:
                    self.ck_state = 'snapshot'

                t_start = time.time()
                self.latest_snapshot = OrderedDict()
                for name, ref in self.destination.items():
                    if name not in self.latest_snapshot:
                        # print('name is {}'.format(name))
                        t1 = time.time()
                        self.latest_snapshot[name] = copy.deepcopy(ref.state_dict())
                        t2 = time.time()
                        # print('t:{}'.format(t2 - t1))
                    else:
                        self.logger.info("Repeated entry for {}".format(name))
                        print('Repeated entry')
                        with lock:
                            self.ck_state = 'idle'
                        return False
                t_stop = time.time() - t_start
                with lock:
                    self.ck_state = 'idle'
                print("a snapshot on GPU takes {}".format(t_stop))

                return True
            else:
                return False
        except:
            print('Program exception when snapshot on GPU.')

    def _snapshot_CPU(self,
                      lock,
                      meta_data):

        def _to_cpu(ele, snapshot=None):
            if snapshot is None:
                snapshot = {}
            if hasattr(ele, 'cpu'):
                snapshot = ele.cpu()
            elif isinstance(ele, dict):
                snapshot = {}
                for k, v in ele.items():
                    snapshot[k] = None
                    snapshot[k] = _to_cpu(v, snapshot[k])
            elif isinstance(ele, list):
                snapshot = [None for _ in range(len(ele))]
                for idx, v in enumerate(ele):
                    snapshot[idx] = _to_cpu(v, snapshot[idx])
            return snapshot

        try:
            if self.ck_state == 'idle':
                with lock:
                    self.ck_state = 'snapshot'

                self.latest_snapshot = OrderedDict()
                snapshot_copy_CPU = OrderedDict()
                # print('ck state is {} and time is {}'.format(self.ck_state, time.time()))
                t_start = time.time()
                for name, ref in self.destination.items():
                    if name not in self.latest_snapshot:
                        if name == 'model':
                            self.latest_snapshot[name] = _to_cpu(ref.state_dict())
                            snapshot_copy_CPU[name] = copy.deepcopy(self.latest_snapshot[name])
                        elif name == 'optimizer':
                            self.latest_snapshot[name] = copy.deepcopy(ref.state_dict())
                    else:
                        self.logger.info("Repeated entry for {}".format(name))
                        return False
                t_stop = time.time() - t_start
                print("a snapshot on CPU takes {}s".format(t_stop))
                # print('ck state is {} and time is {}'.format(self.ck_state, time.time()))

                with lock:
                    self.ck_state = 'idle'

                return True
            else:
                return False
        except:
            print('Program exception when snapshot on CPU.')

    def _persist(self,
                 ck_filepath,
                 meta_filepath,
                 snapshot,
                 meta_state,
                 lock,
                 ckname_dict:OrderedDict):
        # self.logger.info('persist begin!!')
        try:
            t_start = time.time()
            if self.ck_state == 'idle':
                with lock:
                    self.ck_state = 'persist'

                torch.save(snapshot, ck_filepath)
                f = open(ck_filepath, 'a+')
                os.fsync(f.fileno())
                f.close()

                if not isinstance(meta_state, Mapping):
                    self.logger.info("meta_state is not Mapping.")
                    return False

                # print('meta state is {}'.format(meta_state))
                js = json.dumps(meta_state)
                f = open(meta_filepath, 'w+')
                f.write(js)
                f.close()

                self.ck_complete = True

                if ckname_dict is not None:
                    fname = os.path.splitext(os.path.basename(ck_filepath))[0]
                    if fname not in ckname_dict.keys():
                        ckname_dict[fname] = self.ck_complete

                    if len(ckname_dict) > 2:
                        del_filepath = os.path.join(os.path.dirname(ck_filepath), list(ckname_dict.keys())[0] + '.chk')
                        if os.path.exists(del_filepath):
                            os.remove(del_filepath)

                        del_filepath = os.path.join(os.path.dirname(meta_filepath), list(ckname_dict.keys())[0] + '.meta')
                        if os.path.exists(del_filepath):
                            os.remove(del_filepath)

                        del ckname_dict[list(ckname_dict.keys())[0]]

                with lock:
                    self.ck_state = 'idle'

                t_stop = time.time() - t_start
                print("persisting a snapshot takes {}s".format(t_stop))
                return True
            else:
                return False
        except:
            print('Program exception when persist.')

    def _pipeline_ck_GPU(self,
                         lock,
                         meta_state,
                         ck_filepath,
                         meta_filepath,
                         ckname_dict):
        flag = self._snapshot_GPU(lock, meta_state)
        if flag is True:
            self._persist(ck_filepath=ck_filepath,
                          meta_filepath=meta_filepath,
                          snapshot=self.latest_snapshot,
                          meta_state=meta_state,
                          lock=lock,
                          ckname_dict=ckname_dict)

    def _pipeline_ck_CPU(self,
                         lock,
                         meta_state,
                         ck_filepath,
                         meta_filepath,
                         ckname_dict):

        flag = self._snapshot_CPU(lock, meta_state)
        if flag is True:
            self._persist(ck_filepath=ck_filepath,
                          meta_filepath=meta_filepath,
                          snapshot=self.latest_snapshot,
                          meta_state=meta_state,
                          lock=lock,
                          ckname_dict=ckname_dict)

    def _recovery(self,
                  filepath,
                  gpu=0):
        # load all of tensors into device gpu=0
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage.cuda(gpu))

        # reinitialize state of tractable objects
        self.destination['model'].load_state_dict(checkpoint['model'])
        self.destination['optimizer'].load_state_dict(checkpoint['optimizer'])
        # for name, ref in self.destination.items():
        #     try:
        #         ref.load_state_dict(checkpoint[name])
        #         del checkpoint[name]
        #     except ValueError:
        #         print("Corrupt checkpoint")

        if len(checkpoint.keys()) > 0:
            return True
        else:
            print('checkpoint is none')
            return False

