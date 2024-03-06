# https://github.com/ContinualAI/avalanche/blob/master/avalanche/evaluation/metrics/mean.py
# https://github.com/ContinualAI/avalanche/blob/master/avalanche/evaluation/metrics/cpu_usage.py

import os

from psutil import Process


class Mean:
    def __init__(self):
        """
        Creates an instance of the mean metric.

        This metric in its initial state will return a mean value of 0.
        The metric can be updated by using the `update` method while the mean
        can be retrieved using the `result` method.
        """
        self.summed: float = 0.0
        self.weight: float = 0.0

    def update(self, value, weight=1.0) -> None:
        """
        Update the running mean given the value.

        The value can be weighted with a custom value, defined by the `weight`
        parameter.

        :param value: The value to be used to update the mean.
        :param weight: The weight of the value. Defaults to 1.
        :return: None.
        """
        value = float(value)
        weight = float(weight)
        self.summed += value * weight
        self.weight += weight

    def result(self) -> float:
        """
        Retrieves the mean.

        Calling this method will not change the internal state of the metric.

        :return: The mean, as a float.
        """
        if self.weight == 0.0:
            return 0.0
        return self.summed / self.weight

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """
        self.summed = 0.0
        self.weight = 0.0

    def __add__(self, other: "Mean") -> "Mean":
        """
        Return a metric representing the weighted mean of the 2 means.

        :param other: the other mean
        :return: The weighted mean"""
        res = Mean()
        res.summed = self.summed + other.summed
        res.weight = self.weight + other.weight
        return res


class CPUUsage:
    """
    The standalone CPU usage metric.

    Instances of this metric compute the average CPU usage as a float value.
    The metric starts tracking the CPU usage when the `update` method is called
    for the first time. That is, the tracking does not start at the time the
    constructor is invoked.

    Calling the `update` method more than twice will update the metric to the
    average usage between the first and the last call to `update`.

    The result, obtained using the `result` method, is the usage computed
    as stated above.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an usage value of 0.
    """

    def __init__(self):
        """
        Creates an instance of the standalone CPU usage metric.

        By default this metric in its initial state will return a CPU usage
        value of 0. The metric can be updated by using the `update` method
        while the average CPU usage can be retrieved using the `result` method.
        """

        self._mean_usage = Mean()
        """
        The mean utility that will be used to store the average usage.
        """

        self._process_handle: None
        """
        The process handle, lazily initialized.
        """

        self._first_update = True
        """
        An internal flag to keep track of the first call to the `update` method.
        """

    def update(self) -> None:
        """
        Update the running CPU usage.

        For more info on how to set the starting moment see the class
        description.

        :return: None.
        """
        if self._first_update:
            self._process_handle = Process(os.getpid())

        assert self._process_handle is not None

        last_time = getattr(self._process_handle, "_last_sys_cpu_times", None)
        utilization = self._process_handle.cpu_percent()
        current_time = getattr(self._process_handle, "_last_sys_cpu_times", None)

        if self._first_update:
            self._first_update = False
        else:
            self._mean_usage.update(utilization, current_time - last_time)

    @property
    def result(self) -> float:
        """
        Retrieves the average CPU usage.

        Calling this method will not change the internal state of the metric.

        :return: The average CPU usage, as a float value.
        """
        return self._mean_usage.result()

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """
        self._mean_usage.reset()
        self._process_handle = None
        self._first_update = True


__all__ = ["CPUUsage", "Mean"]
