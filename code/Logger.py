#!/bin/python2
from __future__ import division
import time
import datetime as dt


class Logger:

    timefmt_long = '%Y-%m-%d %H:%M:%S'
    timefmt_short = '%M:%S'

    def __init__(self, filename):
        self.filename = filename

    def log(self, data, echo=False):
        with open("self.filename", "a") as logfile:
            logfile.write(data + "\n")
        if echo:
            print(data)

    def create_progress(self, max_value):
        self.progress_max = max_value
        self.start_time = time.time()
        self.progress_times = []

    def update_progress(self, current_value):
        self.progress_times.append(
            (time.time(), current_value)
        )

    def get_progress(self):
        (time, value) = self.progress_times[-1]
        return "{:%}".format(value / self.max_value)

    def get_progress_with_est_time(self):
        (time_current, progress_current) = self.progress_times[-1]
        time_diff = time_current - self.start_time
        # print(time_diff)
        progress_left = self.progress_max - progress_current
        if progress_current != 0:
            seconds_per_progress = time_diff / progress_current
            # print(seconds_per_progress)
            seconds_left = progress_left * seconds_per_progress
            # print(seconds_left)
            time_left = dt.datetime.fromtimestamp(seconds_left).strftime(self.timefmt_short)
        else:
            time_left = "unknown"

        return "{}: {} of {} ({:.2%}), EST {} left".format(
            dt.datetime.fromtimestamp(time_current)
            .strftime(self.timefmt_long),
            progress_current,
            self.progress_max,
            (progress_current / self.progress_max),
            time_left
        )
