import re
from dataclasses import dataclass, field
from pathlib import PosixPath, PurePath
from typing import Any, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import pysftp
from snakemake_interface_storage_plugins.io import IOCacheStorageInterface  # , get_constant_prefix
from snakemake_interface_storage_plugins.settings import StorageProviderSettingsBase
from snakemake_interface_storage_plugins.storage_object import StorageObjectGlob, StorageObjectRead, StorageObjectWrite, retry_decorator
from snakemake_interface_storage_plugins.storage_provider import ExampleQuery, Operation, QueryType, StorageProviderBase, StorageQueryValidationResult
from snakemake_storage_plugin_sftp.cnopts import CnOpts


# Optional:
# Define settings for your storage plugin (e.g. host url, credentials).
# They will occur in the Snakemake CLI as --storage-<storage-plugin-name>-<param-name>
# Make sure that all defined fields are 'Optional' and specify a default value
# of None or anything else that makes sense in your case.
# Note that we allow storage plugin settings to be tagged by the user. That means,
# that each of them can be specified multiple times (an implicit nargs=+), and
# the user can add a tag in front of each value (e.g. tagname1:value1 tagname2:value2).
# This way, a storage plugin can be used multiple times within a workflow with different
# settings.
@dataclass
class StorageProviderSettings(StorageProviderSettingsBase):
    username: Optional[str] = field(
        default=None,
        metadata={
            "help": "SFTP username",
            # Optionally request that setting is also available for specification
            # via an environment variable. The variable will be named automatically as
            # SNAKEMAKE_<storage-plugin-name>_<param-name>, all upper case.
            # This mechanism should only be used for passwords, usernames, and other
            # credentials.
            # For other items, we rather recommend to let people use a profile
            # for setting defaults
            # (https://snakemake.readthedocs.io/en/stable/executing/cli.html#profiles).
            "env_var": True,
            # Optionally specify that setting is required when the executor is in use.
            "required": False,
        },
    )
    password: Optional[str] = field(
        default=None,
        metadata={
            "help": "SFTP password",
            # Optionally request that setting is also available for specification
            # via an environment variable. The variable will be named automatically as
            # SNAKEMAKE_<storage-plugin-name>_<param-name>, all upper case.
            # This mechanism should only be used for passwords, usernames, and other
            # credentials.
            # For other items, we rather recommend to let people use a profile
            # for setting defaults
            # (https://snakemake.readthedocs.io/en/stable/executing/cli.html#profiles).
            "env_var": True,
            # Optionally specify that setting is required when the executor is in use.
            "required": False,
        },
    )
    not_sync_mtime: bool = field(
        default=False,
        metadata={
            "help": "Do not synchronize mtime when storing files or dirs.",
        },
    )


# Required:
# Implementation of your storage provider
# This class can be empty as the one below.
# You can however use it to store global information or maintain e.g. a connection
# pool.
class StorageProvider(StorageProviderBase):
    # For compatibility with future changes, you should not overwrite the __init__
    # method. Instead, use __post_init__ to set additional attributes and initialize
    # further stuff.

    def __post_init__(self):
        # This is optional and can be removed if not needed.
        # Alternatively, you can e.g. prepare a connection to your storage backend here.
        # and set additional attributes.
        self.conn_pool = dict()

    @classmethod
    def example_queries(cls) -> List[ExampleQuery]:
        """Return an example query with description for this storage provider."""
        return [
            ExampleQuery(
                query="sftp://ftpserver.com:22/myfile.txt",
                type=QueryType.ANY,
                description="A file on an sftp server. " "The port is optional and defaults to 22.",
            )
        ]

    def use_rate_limiter(self) -> bool:
        """Return False if no rate limiting is needed for this provider."""
        return True

    def default_max_requests_per_second(self) -> float:
        """Return the default maximum number of requests per second for this storage
        provider."""
        return 1.0

    def rate_limiter_key(self, query: str, operation: Operation):
        """Return a key for identifying a rate limiter given a query and an operation.

        This is used to identify a rate limiter for the query.
        E.g. for a storage provider like http that would be the host name.
        For s3 it might be just the endpoint URL.
        """
        parsed = urlparse(query)
        return parsed.netloc

    @classmethod
    def is_valid_query(cls, query: str) -> StorageQueryValidationResult:
        """Return whether the given query is valid for this storage provider."""
        # Ensure that also queries containing wildcards (e.g. {sample}) are accepted
        # and considered valid. The wildcards will be resolved before the storage
        # object is actually used.
        parsed = urlparse(query)
        if parsed.scheme == "sftp" and parsed.path:
            return StorageQueryValidationResult(valid=True, query=query)
        else:
            return StorageQueryValidationResult(
                valid=False,
                query=query,
                reason="Query does not start with sftp:// or does not contain a path " "to a file or directory.",
            )

    def list_objects(self, query: Any) -> Iterable[str]:
        """Return an iterator over all objects in the storage that match the query.

        This is optional and can raise a NotImplementedError() instead.
        """
        # TODO implement this
        raise NotImplementedError()

    def get_conn(self, hostname: str, port: Optional[int] = 22):
        key = hostname, port
        if key not in self.conn_pool:
            conn = pysftp.Connection(
                hostname,
                port=port,
                cnopts=CnOpts(port=port),
                username=self.settings.username,
                password=self.settings.password,
            )
            self.conn_pool[key] = conn
            return conn
        return self.conn_pool[key]


WILDCARD_REGEX = re.compile(
    r"""
    \{
        (?=(   # This lookahead assertion emulates an 'atomic group'
               # which is required for performance
            \s*(?P<name>\w+)                    # wildcard name
            (\s*,\s*
                (?P<constraint>                 # an optional constraint
                    ([^{}]+ | \{\d+(,\d+)?\})*  # allow curly braces to nest one level
                )                               # ...  as in '{w,a{3,5}}'
            )?\s*
        ))\1
    \}
    """,
    re.VERBOSE,
)


def get_prefix(pattern: str, strip_incomplete_parts: bool = False) -> Tuple[str, str]:
    """Return constant prefix of a pattern, removing everything from the first
    wildcard on.

    If strip_incomplete_parts is set, trailing parts that do not end with
    a slash (/) are removed as well.
    """
    first_wildcard = WILDCARD_REGEX.search(pattern)
    if first_wildcard:
        constant_prefix = pattern[: first_wildcard.start()]
        if strip_incomplete_parts:
            if "/" in constant_prefix:
                constant_prefix = constant_prefix.rsplit("/", 1)[0] + "/"
            else:
                first_slash_idx = pattern.find("/")
                if first_slash_idx != -1 and first_slash_idx > first_wildcard.start():
                    # the first slash is after the first wildcard, hence the prefix
                    # is incomplete
                    constant_prefix = ""
    else:
        constant_prefix = pattern

    # find last constant_prefix part in pattern
    last_idx = pattern.rfind(constant_prefix[-1])
    if last_idx != -1:
        glob_prefix = pattern[last_idx + 1 :]
        # convert {wildcard} to *
        glob_prefix = WILDCARD_REGEX.sub("*", glob_prefix)
        return constant_prefix, glob_prefix

    return constant_prefix, ""

    # query_p = PurePath(query)
    # # constant_prefix, glob_prefix = "", ""
    # i = 0
    # for p in query_p.parts:
    #     if "*" in p or "?" in p:
    #         break
    #     i += 1
    # return "/".join([str(p) for p in query_p.parts[:i]]), "/".join([str(p) for p in query_p.parts[i:]])


# Required:
# Implementation of storage object. If certain methods cannot be supported by your
# storage (e.g. because it is read-only see
# snakemake-storage-http for comparison), remove the corresponding base classes
# from the list of inherited items.
class StorageObject(StorageObjectRead, StorageObjectWrite, StorageObjectGlob):
    # For compatibility with future changes, you should not overwrite the __init__
    # method. Instead, use __post_init__ to set additional attributes and initialize
    # further stuff.

    def __post_init__(self):
        # This is optional and can be removed if not needed.
        # Alternatively, you can e.g. prepare a connection to your storage backend here.
        # and set additional attributes.
        self.parsed_query = urlparse(self.query)
        self.conn = self.provider.get_conn(self.parsed_query.hostname, self.parsed_query.port)

    async def inventory(self, cache: IOCacheStorageInterface):
        """From this file, try to find as much existence and modification date
        information as possible. Only retrieve that information that comes for free
        given the current object.
        """
        # This is optional and can be left as is

        # If this is implemented in a storage object, results have to be stored in
        # the given IOCache object.
        # TODO implement this
        pass

    def get_inventory_parent(self) -> Optional[str]:
        """Return the parent directory of this object."""
        # this is optional and can be left as is
        return None

    def local_suffix(self) -> str:
        """Return a unique suffix for the local path, determined from self.query."""
        # This is optional and can be left as is
        return f"{self.parsed_query.netloc}/{self.parsed_query.path}"

    def cleanup(self):
        """Perform local cleanup of any remainders of the storage object."""
        # self.local_path() should not be removed, as this is taken care of by
        # Snakemake.
        pass

    # Fallible methods should implement some retry logic.
    # The easiest way to do this (but not the only one) is to use the retry_decorator
    # provided by snakemake-interface-storage-plugins.
    @retry_decorator
    def exists(self) -> bool:
        # return True if the object exists
        return self.conn.exists(self.parsed_query.path)

    @retry_decorator
    def mtime(self) -> float:
        # return the modification time
        return self.conn.lstat(self.parsed_query.path).st_mtime

    @retry_decorator
    def size(self) -> int:
        # return the size in bytes
        return self.conn.stat(self.parsed_query.path).st_size

    @retry_decorator
    def retrieve_object(self):
        # Ensure that the object is accessible locally under self.local_path()
        get = self.conn.get_r if self.conn.isdir(self.parsed_query.path) else self.conn.get
        get(self.parsed_query.path, self.local_path(), preserve_mtime=True)

    # The following to methods are only required if the class inherits from
    # StorageObjectReadWrite.

    @retry_decorator
    def store_object(self):
        # Ensure that the object is stored at the location specified by
        # self.local_path().
        put = self.conn.put_r if self.local_path().is_dir() else self.conn.put
        parents = str(PosixPath(self.parsed_query.path).parent)
        if parents != ".":
            self.conn.makedirs(parents)
        put(
            self.local_path(),
            self.parsed_query.path,
            preserve_mtime=not self.provider.settings.not_sync_mtime,
            confirm=True,
        )

    @retry_decorator
    def remove(self):
        # Remove the object from the storage.
        remove = self.conn.rmdir if self.conn.isdir(self.parsed_query.path) else self.conn.remove
        remove(self.parsed_query.path)

    # The following to methods are only required if the class inherits from
    # StorageObjectGlob.

    @retry_decorator
    def list_candidate_matches(self) -> Iterable[str]:
        """Return a list of candidate matches in the storage for the query."""
        # This is used by glob_wildcards() to find matches for wildcards in the query.
        # The method has to return concretized queries without any remaining wildcards.

        # old_prefix = get_constant_prefix(self.query, strip_incomplete_parts=True)
        # query_p = PurePath(self.query)
        # # prefix is the part of the query that is constant and does not contain any wildcards
        # constant_prefix = "/".join([str(p) for p in query_p.parts if "*" not in p and "?" not in p])
        # print(f"constant_prefix: {constant_prefix}")
        # url = urlparse(constant_prefix)
        # prefix = constant_prefix.replace(f"{url.scheme}://{url.netloc}", "")
        # is_glob = "*" in prefix or "?" in prefix
        # glob_prefix = query_p.stem
        # # print(f"prefix: {prefix}")
        # if is_glob:
        #     prefix = get_prefix_without_glob(prefix)

        constant_prefix, glob_prefix = get_prefix(self.query)
        url = urlparse(constant_prefix)
        prefix = constant_prefix.replace(f"{url.scheme}://{url.netloc}", "")
        prefix_p = PosixPath(prefix)
        is_glob = "*" in glob_prefix or "?" in glob_prefix
        glob_prefix = str(PosixPath(prefix) / glob_prefix)
        # print(f"constant_prefix: {constant_prefix}, glob_prefix: {glob_prefix}")
        # print(f"prefix: {prefix}, is_glob: {is_glob}, glob_prefix_p: {glob_prefix_p}")

        items = []
        if self.conn.isdir(prefix):
            prefix_p = PosixPath(prefix)

            def yieldfile(path):
                p = prefix_p / path
                p_str = str(p)
                # print(f"file, p_str: {p_str}, match: {glob_prefix_p.match(p_str)}")
                if not is_glob or (is_glob and p.match(glob_prefix)):
                    items.append(p_str.replace(prefix, constant_prefix))

            def yielddir(path):
                # only yield directories that are empty
                if not self.conn.listdir(str(prefix_p / path)):
                    p = prefix_p / path
                    p_str = str(p)
                    # print(f"dir, p_str: {p_str}")
                    if not is_glob or (is_glob and p.match(glob_prefix)):
                        items.append(p_str.replace(prefix, constant_prefix))

            def yieldunk(_): ...

            self.conn.walktree(prefix, fcallback=yieldfile, dcallback=yielddir, ucallback=yieldunk)
        elif self.conn.exists(prefix):
            items.append(str(constant_prefix))
        print("items: ", items)
        return items
