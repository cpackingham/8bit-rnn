from .plugin import Plugin
from librosa.output import write_wav
from matplotlib import pyplot

from glob import glob
import os
import pickle
import time
class StatsPlugin(Plugin):

    data_file_name = 'stats.pkl'
    plot_pattern = '{}.svg'

    def __init__(self, results_path, iteration_fields, epoch_fields, plots):
        super().__init__([(1, 'iteration'), (1, 'epoch')])
        self.results_path = results_path

        self.iteration_fields = self._fields_to_pairs(iteration_fields)
        self.epoch_fields = self._fields_to_pairs(epoch_fields)
        self.plots = plots
        self.data = {
            'iterations': {
                field: []
                for field in self.iteration_fields + [('iteration', 'last')]
            },
            'epochs': {
                field: []
                for field in self.epoch_fields + [('iteration', 'last')]
            }
        }

    def register(self, trainer):
        print("Registered!")
        print(self.results_path)
        self.trainer = trainer

    def iteration(self, *args):
        for (field, stat) in self.iteration_fields:
            self.data['iterations'][field, stat].append(
                self.trainer.stats[field][stat]
            )

        self.data['iterations']['iteration', 'last'].append(
            self.trainer.iterations
        )

    def epoch(self, epoch_index):
        for (field, stat) in self.epoch_fields:
            self.data['epochs'][field, stat].append(
                self.trainer.stats[field][stat]
            )

        self.data['epochs']['iteration', 'last'].append(
            self.trainer.iterations
        )
        print("saving!")
        print(self.data_file_name)
        data_file_path = os.path.join(self.results_path, self.data_file_name)
        with open(data_file_path, 'wb') as f:
            pickle.dump(self.data, f)

        for (name, info) in self.plots.items():
            x_field = self._field_to_pair(info['x'])

            try:
                y_fields = info['ys']
            except KeyError:
                y_fields = [info['y']]

            labels = list(map(
                lambda x: ' '.join(x) if type(x) is tuple else x,
                y_fields
            ))
            y_fields = self._fields_to_pairs(y_fields)

            try:
                formats = info['formats']
            except KeyError:
                formats = [''] * len(y_fields)

            pyplot.gcf().clear()

            for (y_field, format, label) in zip(y_fields, formats, labels):
                if y_field in self.iteration_fields:
                    part_name = 'iterations'
                else:
                    part_name = 'epochs'

                xs = self.data[part_name][x_field]
                ys = self.data[part_name][y_field]

                pyplot.plot(xs, ys, format, label=label)

            if 'log_y' in info and info['log_y']:
                pyplot.yscale('log')

            pyplot.legend()
            pyplot.savefig(
                os.path.join(self.results_path, self.plot_pattern.format(name))
            )

    @staticmethod
    def _field_to_pair(field):
        if type(field) is tuple:
            return field
        else:
            return (field, 'last')

    @classmethod
    def _fields_to_pairs(cls, fields):
        return list(map(cls._field_to_pair, fields))
