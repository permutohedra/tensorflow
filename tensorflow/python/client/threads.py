"""Operations to pause/resume all threads in TensorFlow ThreadPools.
May be used to safely fork a process with live Session objects.
"""

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.util.tf_export import tf_export

@tf_export('pause_all_threads')
def pause_all_threads():
    """Shut down all worker threads in TensorFlow thread pools, after waiting for work to complete.
    After this call returns, it is safe to fork the process. It is an error to schedule any work on
    a thread pool (e.g. by running ops in a session) before `tf.resume_all_threads()` is called.
    Usage:

    ```python
    tf.pause_all_threads()
    if os.fork():
        os.wait()
    else:
        tf.resume_all_threads()
        session.run(...)
    ```

    Performance tip: warm up the session by running the graph once before pausing threads and
    forking.
    """
    pywrap_tensorflow.TF_PauseAllThreads()

@tf_export('resume_all_threads')
def resume_all_threads():
    """Recreate all threads stopped by `tf.pause_all_threads`.
    Call this after forking and before running any session.
    """
    pywrap_tensorflow.TF_ResumeAllThreads()
