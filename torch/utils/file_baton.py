# mypy: allow-untyped-defs
import os
import time
<<<<<<< HEAD
=======
import warnings
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


class FileBaton:
    """A primitive, file-based synchronization utility."""

<<<<<<< HEAD
    def __init__(self, lock_file_path, wait_seconds=0.1):
=======
    def __init__(self, lock_file_path, wait_seconds=0.1, warn_after_seconds=None):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        """
        Create a new :class:`FileBaton`.

        Args:
            lock_file_path: The path to the file used for locking.
            wait_seconds: The seconds to periodically sleep (spin) when
                calling ``wait()``.
<<<<<<< HEAD
=======
            warn_after_seconds: The seconds to wait before showing
                lock file path to warn existing lock file.
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        """
        self.lock_file_path = lock_file_path
        self.wait_seconds = wait_seconds
        self.fd = None
<<<<<<< HEAD
=======
        self.warn_after_seconds = warn_after_seconds
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

    def try_acquire(self):
        """
        Try to atomically create a file under exclusive access.

        Returns:
            True if the file could be created, else False.
        """
        try:
            self.fd = os.open(self.lock_file_path, os.O_CREAT | os.O_EXCL)
            return True
        except FileExistsError:
            return False

    def wait(self):
        """
        Periodically sleeps for a certain amount until the baton is released.

        The amount of time slept depends on the ``wait_seconds`` parameter
        passed to the constructor.
        """
<<<<<<< HEAD
        while os.path.exists(self.lock_file_path):
            time.sleep(self.wait_seconds)

=======
        has_warned = False

        start_time = time.time()
        while os.path.exists(self.lock_file_path):
            time.sleep(self.wait_seconds)

            if self.warn_after_seconds is not None:
                if time.time() - start_time > self.warn_after_seconds and not has_warned:
                    warnings.warn(f'Waited on lock file "{self.lock_file_path}" for '
                                  f'{self.warn_after_seconds} seconds.')
                    has_warned = True

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    def release(self):
        """Release the baton and removes its file."""
        if self.fd is not None:
            os.close(self.fd)

        os.remove(self.lock_file_path)
