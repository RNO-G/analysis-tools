import mattak.Dataset as Dataset

import numpy as np
from collections import defaultdict

class Datasets(object):
    def __init__(self, dataset_paths, read_in_init=False, **kwargs):
        self.dataset_paths = dataset_paths
        self.station = None
        self._kwargs = kwargs
        self.runs = []
        self._datasets = {}
        self.num_events = 0
        self.read_in_init = read_in_init

        if read_in_init:
            for path in dataset_paths:
                dataset = self.open_dataset(path)
                dataset.setEntries((0, dataset.N()))
                self.datasets[(self.num_events, self.num_events + dataset.N())] = dataset
                self.num_events += dataset.N()


    def open_dataset(self, path):
        dataset = Dataset.Dataset(station=0, run=0, data_path=path, **self._kwargs)
        self.runs.append(dataset.run)
        if self.station is None:
            self.station = dataset.station
        else:
            # this could be changed in the future if there is a usecase for it
            assert self.station == dataset.station, "All datasets must be from the same station"

        return dataset

    def N(self):
        return self.num_events

    def wfs(self, **kwargs):
        if self.read_in_init:
            return np.hstack([dataset.wfs(**kwargs) for dataset in self.datasets.values()])
        else:
            wfs = []
            for path in self.dataset_paths:
                dataset = self.open_dataset(path)
                wfs.append(dataset.wfs(**kwargs))

            return np.hstack(wfs)

    def eventInfo(self):
        if self.read_in_init:
            return np.hstack([dataset.eventInfo() for dataset in self.datasets.values()])
        else:
            event_info = []
            for path in self.dataset_paths:
                dataset = self.open_dataset(path)
                event_info.append(dataset.eventInfo())

            return np.hstack(event_info)

    def iterate(self, **kwargs):
        for dataset in self.datasets.values():
            dataset.setEntries(0, dataset.N())
            for ev, wfs in dataset.iterate(**kwargs):
                yield ev, wfs


def convert_events_information(event_info):

    data = defaultdict(list)

    for ele in event_info.values():
        for k, v in ele.items():
            data[k].append(v)

    for k in data:
        data[k] = np.array(data[k])

    return data
