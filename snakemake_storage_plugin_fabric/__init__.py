import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, List, Literal, Optional
from urllib.parse import urlparse

import fabric
from snakemake_interface_storage_plugins.io import IOCacheStorageInterface, get_constant_prefix
from snakemake_interface_storage_plugins.settings import StorageProviderSettingsBase
from snakemake_interface_storage_plugins.storage_object import StorageObjectGlob, StorageObjectRead, StorageObjectWrite, retry_decorator
from snakemake_interface_storage_plugins.storage_provider import ExampleQuery, Operation, QueryType, StorageProviderBase, StorageQueryValidationResult


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
            "help": "SFTP/SSH username",
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
            "help": "SFTP/SSH password",
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


def isdir(conn: fabric.Connection, path: str | Path) -> bool:
    # cmd = f"ls -la {path} | grep '^d'"
    # run ls -la and get first character of the first line
    # if it is a d, it is a directory
    cmd = f"ls -la {path} | head -n 1"
    str = conn.run(cmd, hide=True).stdout
    return str[0] == "d"


def exists(conn: fabric.Connection, path: str | Path) -> bool:
    cmd = f"ls {path}"
    return conn.run(cmd, hide=True).ok


def listdir(conn: fabric.Connection, path: str | Path) -> List[str]:
    return conn.run(f"ls {path}", hide=True).stdout.splitlines()


def walktree(conn: fabric.Connection, path: str | Path, fcallback: Callable[[str | Path], None], dcallback: Callable[[str | Path], None]):
    for p in listdir(conn, path):
        p = Path(p)
        if isdir(conn, path / p):
            dcallback(p)
            walktree(conn, path / p, fcallback, dcallback)
        else:
            fcallback(p)


def size(conn: fabric.Connection, path: str) -> int:
    return conn.sftp().stat(path).st_size


def mtime(conn: fabric.Connection, path: str) -> float:
    return conn.sftp().stat(path).st_mtime


def mkdir(conn: fabric.Connection, path: str):
    conn.run(f"mkdir -p {path}")


def remove(conn: fabric.Connection, path: str):
    if isdir(conn, path):
        conn.run(f"rm -rf {path}")
    else:
        conn.run(f"rm {path}")


def rsync(
    conn: fabric.Connection,
    src: str | Path,
    dest: str | Path,
    exclude: List[str] = [],
    delete: bool = False,
    dry_run: bool = False,
    rsync_opts: str = "",
    ssh_opts: str = "",
    strict_host_key_checking: bool = False,
    action: Literal["push", "pull"] = "push",
):
    if action == "pull":
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
    else:
        ...
        # TODO: make sure the destination directory exists in the remote host

    exclude_opts = ' --exclude "{}"' * len(exclude)
    # Double-backslash-escape
    exclusions = tuple([str(s).replace('"', '\\\\"') for s in exclude])
    key_string = ""
    keys = conn.connect_kwargs.get("key_filename", []) if conn.connect_kwargs else []
    if isinstance(keys, str):
        keys = [keys]
    if keys:
        key_string = " -i " + " -i ".join(keys)
    # if dry_run:
    #     rsync_opts += "n"
    # if delete:
    #     rsync_opts += "r"
    # if exclude:
    #     rsync_opts += " --exclude " + " --exclude ".join(exclude)
    user, host, port = conn.user, conn.host, conn.port
    port_string = f"-p {port}" if port else ""

    rsh_string = ""
    disable_keys = "-o StrictHostKeyChecking=no"
    if not strict_host_key_checking and disable_keys not in ssh_opts:
        ssh_opts += " " + disable_keys
    rsh_parts = [key_string, port_string, ssh_opts]
    if any(rsh_parts):
        rsh_string = "--rsh='ssh {}'".format(" ".join(rsh_parts))
    options_map = {
        "dry_run": "--dry-run" if dry_run else "",
        "delete": "--delete" if delete else "",
        "exclude": exclude_opts.format(*exclusions),
        "extra": rsync_opts,
        "rsh": rsh_string,
    }
    options = "{dry_run} {delete}{exclude} -pthrvz {extra} {rsh}".format(**options_map)
    if action == "push":
        if host and host.count(":") > 1:
            # Square brackets are mandatory for IPv6 rsync address,
            # even if port number is not specified
            cmd = "rsync {} {} [{}@{}]:{}"
        else:
            cmd = "rsync {} {} {}@{}:{}"
        cmd = cmd.format(options, src, user, host, dest)
    else:
        if host and host.count(":") > 1:
            # Square brackets are mandatory for IPv6 rsync address,
            # even if port number is not specified
            cmd = "rsync {} [{}@{}]:{} {}"
        else:
            cmd = "rsync {} {}@{}:{} {}"
        cmd = cmd.format(options, user, host, src, Path(dest).parent)
    return conn.local(cmd)


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
        if (parsed.scheme == "sftp" or parsed.scheme == "ssh") and parsed.path:
            return StorageQueryValidationResult(valid=True, query=query)
        else:
            return StorageQueryValidationResult(
                valid=False,
                query=query,
                reason="Query does not start with sftp:// or ssh:// or does not contain a path " "to a file or directory.",
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
            connect_kwargs = {}
            if self.settings.password:  # type: ignore
                connect_kwargs["password"] = self.settings.password  # type: ignore
            conn = fabric.Connection(
                hostname,
                port=port,
                user=self.settings.username,  # type: ignore
                connect_kwargs=connect_kwargs,
            )
            self.conn_pool[key] = conn
            return conn
        return self.conn_pool[key]


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
        self.conn: fabric.Connection = self.provider.get_conn(self.parsed_query.hostname, self.parsed_query.port)  # type: ignore

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

    def local_suffix(self) -> str:  # type: ignore
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
        return exists(self.conn, self.parsed_query.path)

    @retry_decorator
    def mtime(self) -> float:
        # return the modification time
        return mtime(self.conn, self.parsed_query.path)

    @retry_decorator
    def size(self) -> int:
        # return the size in bytes
        return size(self.conn, self.parsed_query.path)

    @retry_decorator
    def retrieve_object(self):
        # print(f"Retrieving {self.parsed_query.path} to {self.local_path()}")
        # Ensure that the object is accessible locally under self.local_path()
        sftpattrs = None
        if not self.provider.settings.not_sync_mtime:  # type: ignore
            sftpattrs = self.conn.sftp().stat(self.parsed_query.path)

        rsync(self.conn, self.parsed_query.path, self.local_path(), rsync_opts="-a", action="pull")
        # self.conn.get(self.parsed_query.path, str(self.local_path()))
        if sftpattrs is not None and not self.provider.settings.not_sync_mtime:  # type: ignore
            os.utime(self.local_path(), (sftpattrs.st_atime, sftpattrs.st_mtime))

    # The following to methods are only required if the class inherits from
    # StorageObjectReadWrite.

    @retry_decorator
    def store_object(self):
        # Ensure that the object is stored at the location specified by
        # self.local_path().
        # put = self.conn.put_r if self.local_path().is_dir() else self.conn.put
        parents = str(Path(self.parsed_query.path).parent)
        if parents != ".":
            mkdir(self.conn, parents)

        self.conn.put(self.local_path(), self.parsed_query.path)
        if not self.provider.settings.not_sync_mtime:  # type: ignore
            self.conn.sftp().utime(self.parsed_query.path, (self.mtime(), self.mtime()))
            # self.conn.run(f"touch -d @{self.mtime()} {self.parsed_query.path}")

    @retry_decorator
    def remove(self):
        # Remove the object from the storage.
        remove(self.conn, self.parsed_query.path)

    # The following to methods are only required if the class inherits from
    # StorageObjectGlob.

    @retry_decorator
    def list_candidate_matches(self) -> Iterable[str]:
        """Return a list of candidate matches in the storage for the query."""
        # This is used by glob_wildcards() to find matches for wildcards in the query.
        # The method has to return concretized queries without any remaining wildcards.
        prefix = get_constant_prefix(self.query, strip_incomplete_parts=True)
        items: list[str] = []
        if isdir(self.conn, prefix):
            prefix = Path(prefix)

            def yieldfile(path):
                items.append(str(prefix / path))

            def yielddir(path):
                # only yield directories that are empty
                if not listdir(self.conn, str(prefix / path)):
                    items.append(str(prefix / path))

            walktree(self.conn, prefix, fcallback=yieldfile, dcallback=yielddir)
        elif exists(self.conn, prefix):
            items.append(prefix)
        return items
