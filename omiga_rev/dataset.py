"""Replay buffer over one released OMIGA dataset (data/<scenario>-<conf>-<quality>.hdf5), as in the released
datasets/offline_dataset.py: terminal rows are dropped, mask = 1 - d[t+1].

to_device() moves the whole dataset to the GPU once and samples with torch.randint on the device; without it, sample()
draws indices with np.random and copies the minibatch. The two paths use different random streams, so a run is
reproducible only with the same setting (the paper's runs: on the GPU for OMIGA and OMIGA+BC, see train_omiga.py and
omiga_bc.py; on the host for the behaviour cloning and factorial scripts, which draw their own indices).
"""
import numpy as np
import torch
import h5py


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, n_agents, env_name, data_dir, max_size=int(2e6), device='cuda'):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.n_agents = n_agents
        self.env_name = env_name
        self.data_dir = data_dir

        self.o = np.zeros((max_size, n_agents, state_dim))
        self.s = np.zeros((max_size, n_agents, state_dim))
        self.a = np.zeros((max_size, n_agents, action_dim))
        self.r = np.zeros((max_size, 1))
        self.mask = np.zeros((max_size, 1))
        self.s_next = np.zeros((max_size, n_agents, state_dim))
        self.o_next = np.zeros((max_size, n_agents, state_dim))
        self.a_next = np.zeros((max_size, n_agents, state_dim))
        self.device = torch.device(device)

    _FIELDS = ('o', 's', 'a', 'r', 'mask', 's_next', 'o_next', 'a_next')

    def to_device(self):
        if self.device.type != 'cuda':
            return self
        for k in self._FIELDS:
            setattr(self, k, torch.as_tensor(getattr(self, k), dtype=torch.float32,
                                             device=self.device))
        self._on_device = True
        return self

    def sample(self, batch_size):
        if getattr(self, '_on_device', False):
            ind = torch.randint(0, self.o.shape[0], (batch_size,), device=self.device)
            return tuple(getattr(self, k)[ind] for k in self._FIELDS)
        ind = np.random.randint(0, self.o.shape[0], size=batch_size)
        return tuple(torch.FloatTensor(getattr(self, k)[ind]).to(self.device)
                     for k in self._FIELDS)

    def load(self):
        print('==========Data loading==========')
        data_file = self.data_dir + self.env_name + '.hdf5'
        print('Loading from:', data_file)
        f = h5py.File(data_file, 'r')
        s = np.array(f['s'])
        o = np.array(f['o'])
        a = np.array(f['a'])
        r = np.array(f['r'])
        d = np.array(f['d'])
        f.close()

        data_size = s.shape[0]
        nonterminal_steps, = np.where(
            np.logical_and(
                np.logical_not(d[:,0]),
                np.arange(data_size) < data_size - 1))
        print('Found %d non-terminal steps out of a total of %d steps.' % (
            len(nonterminal_steps), data_size))

        self.o = o[nonterminal_steps]
        self.s = s[nonterminal_steps]
        self.a = a[nonterminal_steps]
        self.r = r[nonterminal_steps].reshape(-1, 1)
        self.mask = 1 - d[nonterminal_steps + 1].reshape(-1, 1)
        self.s_next = s[nonterminal_steps + 1]
        self.o_next = o[nonterminal_steps + 1]
        self.a_next = a[nonterminal_steps + 1]
        self.size = self.o.shape[0]
