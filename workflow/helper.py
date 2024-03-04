MSRC_CHUNKS = {
    "hm": 2,
    "mds": 2,
    "prn": 2,
    "proj": 5,
    "prxy": 2,
    "rsrch": 3,
    "src1": 3,
    "src2": 3,
    "stg": 2,
    "ts": 1,
    "usr": 3,
    "wdev": 4,
    "web": 4,
}


def http(storage, url):
    return storage.http(url, keep_local=True)


def get_clio_data(storage, query):
    return storage.clio(f"sftp://box.rs.ht:23/home/{query}", keep_local=True)


def get_flashnet_data(storage, query):
    return storage.flashnet(f"sftp://box.rs.ht:23/home/{query}", keep_local=True)


def get_clio_msrc_data(storage, trace):
    if trace in MSRC_CHUNKS:
        return [get_clio_data(storage, f"msrc/{trace}_{i}.csv.gz") for i in range(MSRC_CHUNKS[trace])]
    return [get_clio_data(storage, f"msrc/{trace}.csv.gz")]
