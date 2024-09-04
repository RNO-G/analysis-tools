import mattak.Dataset as Dataset

import numpy as np


class Datasets(object):
    def __init__(self, dataset_paths, **kwargs):
        self.datasets = {}
        self.station = None
        self.runs = []

        self.num_events = 0
        for path in dataset_paths:
            dataset = Dataset.Dataset(station=0, run=0, data_path=path, **kwargs)
            self.datasets[(self.num_events, self.num_events + dataset.N())] = dataset
            self.num_events += dataset.N()
            self.runs.append(dataset.run)
            if self.station is None:
                self.station = dataset.station
            else:
                # this could be changed in the future if there is a usecase for it
                assert self.station == dataset.station, "All datasets must be from the same station"

    def N(self):
        return self.num_events

    def wfs(self, **kwargs):
        return np.hstack([dataset.wfs(**kwargs) for dataset in self.datasets.values()])

    def eventInfo(self):
        return np.hstack([dataset.eventInfo() for dataset in self.datasets.values()])

    def iterate(self, **kwargs):
        for dataset in self.datasets.values():
            dataset.setEntries(0, dataset.N())
            for ev, wfs in dataset.iterate(**kwargs):
                yield ev, wfs
