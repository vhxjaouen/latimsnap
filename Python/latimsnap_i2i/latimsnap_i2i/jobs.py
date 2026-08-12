"""In-memory background job manager for long-running model transfers.

A ``run_transfer`` request enqueues a job that executes on a worker thread.
Progress/status is stored thread-safely and read by the client via
``transfer_progress``. The produced result is retained until fetched (or the
session expires) and returned by ``transfer_result``.
"""

import logging
import threading
import time
import uuid

log = logging.getLogger("latimsnap_i2i.jobs")


class Job:
    def __init__(self, job_id, session_id, model_id):
        self.job_id = job_id
        self.session_id = session_id
        self.model_id = model_id
        self.status = "queued"      # queued | running | done | error
        self.progress = 0.0
        self.message = ""
        self.result = None          # {result, metadata} on success
        self.error = None
        self.lock = threading.Lock()


class JobManager:
    def __init__(self, max_workers=1):
        self._jobs = {}
        self._lock = threading.Lock()
        self._threads = threading.BoundedSemaphore(max_workers)

    def create(self, session_id, model_id):
        with self._lock:
            job_id = uuid.uuid4().hex
            job = Job(job_id, session_id, model_id)
            self._jobs[job_id] = job
        return job

    def submit(self, job, fn):
        def _run():
            try:
                self._threads.acquire()
                with job.lock:
                    job.status = "running"
                    job.message = "starting"

                def progress(fraction, message=None):
                    with job.lock:
                        job.progress = max(0.0, min(1.0, float(fraction)))
                        if message:
                            job.message = message

                result = fn(progress)
                with job.lock:
                    job.status = "done"
                    job.progress = 1.0
                    job.message = "complete"
                    job.result = result
            except Exception as exc:  # noqa: BLE001
                log.exception("job %s failed", job.job_id)
                with job.lock:
                    job.status = "error"
                    job.error = str(exc)
            finally:
                self._threads.release()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return job

    def get(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, job_id):
        job = self.get(job_id)
        if job is None:
            return None
        with job.lock:
            return {
                "status": job.status,
                "progress": job.progress,
                "message": job.message,
                "error": job.error if job.status == "error" else None,
                "has_result": job.result is not None,
            }

    def take_result(self, job_id):
        job = self.get(job_id)
        if job is None:
            return None
        with job.lock:
            res = job.result
            # Keep a single-copy fetch; mark consumed so we don't re-download.
            job.result = None
            return res

    def clear_session(self, session_id):
        with self._lock:
            dead = [j for j, v in self._jobs.items()
                    if v.session_id == session_id and v.status in ("done", "error")]
            for j in dead:
                del self._jobs[j]
