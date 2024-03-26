# These all numbers should be the same [5, 4]
N_HISTORY = 3
N_FUTURE = 3

# Filtering slow IO
THPT_DROP_RATE = 1.7

# MODEL
HIDDEN_LAYERS = 2
HIDDEN_SIZE = 64


FEATURE_COLUMNS = [
    "size",
    "queue_len",
    "prev_queue_len_1",
    "prev_queue_len_2",
    "prev_queue_len_3",
    "prev_latency_1",
    "prev_latency_2",
    "prev_latency_3",
    "prev_throughput_1",
    "prev_throughput_2",
    "prev_throughput_3",
]

__all__ = [
    "N_HISTORY",
    "N_FUTURE",
    "FEATURE_COLUMNS",
    "HIDDEN_LAYERS",
    "HIDDEN_SIZE",
]
