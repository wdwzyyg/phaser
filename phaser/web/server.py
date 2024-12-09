import asyncio
from collections import deque
import logging
import random
import multiprocessing
import weakref
import typing as t

from quart import Quart, url_for

from phaser.plan import ReconsPlan

from .types import (
    JobID, WorkerID,
    # worker - server communication
    WorkerMessage, JobResponse, CancelResponse, OkResponse, ServerResponse,
    # server - client communication
    WorkerStatus, WorkerState, WorkerUpdate, WorkersUpdate, LogRecord,
    JobStatus, JobState, JobStatusChange, JobUpdate, LogUpdate, JobStopped, JobMessage, JobsUpdate,
)

T = t.TypeVar('T')


class Subscribable(t.Generic[T]):
    def __init__(self):
        self.subscribers: set[asyncio.Queue[T]] = set()

    async def subscribe(self) -> t.AsyncIterator[T]:
        connection: asyncio.Queue[T] = asyncio.Queue()
        self.subscribers.add(connection)
        try:
            while True:
                yield await connection.get()
        finally:  # called when future is cancelled
            self.subscribers.remove(connection)

    async def message_subscribers(self, msg: T):
        for subscriber in self.subscribers:
            await subscriber.put(msg)


class Worker(Subscribable[WorkerUpdate]):
    def __init__(self, worker_id: WorkerID):
        super().__init__()
        self.status: WorkerStatus = 'queued'
        self.id: WorkerID = worker_id

        self.current_job: t.Optional[weakref.ref[Job]] = None

    def state(self) -> WorkerState:
        return WorkerState(self.id, self.status, {
            'shutdown': url_for('shutdown_worker', worker_id=self.id),
        })

    async def cancel(self):
        await self.set_status('stopping')

    def should_cancel(self) -> bool:
        return self.status == 'stopping'

    async def set_status(self, status: WorkerStatus):
        self.status = status
        await self.message_subscribers(WorkerUpdate(self.id, status))

        if status == 'stopped':
            if self.current_job and (job := self.current_job()):
                await job.set_status('stopped')
            server.workers.schedule_for_removal(self.id, 5.0)

    async def handle_message(self, msg: WorkerMessage) -> ServerResponse:
        if msg.msg == 'shutdown':
            await self.set_status('stopped')
            return OkResponse()

        if self.status in ('queued', 'starting'):
            await self.set_status('idle')

        if (job_id := getattr(msg, 'job_id', None)):
            job = server.jobs[job_id]
            await job.handle_update(msg)

            if job.should_cancel():
                return CancelResponse(self.should_cancel())

        if self.should_cancel():
            return CancelResponse(True)

        if msg.msg in ('poll', 'job_result'):
            if len(server.job_queue):
                # send a new job if available
                job = server.job_queue.popleft()
                self.current_job = weakref.ref(job)
                await self.set_status('running')
                await job.set_status('starting')
                return JobResponse(job.id, job.plan)
            else:
                # otherwise don't
                self.current_job = None
                await self.set_status('idle')

        return OkResponse()

    async def finalize(self):
        pass


class LocalWorker(Worker):
    def __init__(self, worker_id: WorkerID, url: str):
        from phaser.web.worker import run_worker

        super().__init__(worker_id)

        self.process = multiprocessing.Process(target=run_worker, args=[url], daemon=True)
        self.status = 'starting'
        self.process.start()

        self._fut: asyncio.Task[None] = asyncio.create_task(asyncio.to_thread(self._watch_process))

    def _watch_process(self):
        self.process.join()
        # TODO call set_status here
        self.status = 'stopped'

    async def finalize(self):
        self.process.terminate()
        await self._fut


class ManualWorker(Worker):
    def __init__(self, worker_id: WorkerID):
        url = server.get_worker_url(worker_id)
        print(f"Worker command: python -m phaser worker {url}")
        super().__init__(worker_id)


class Workers(Subscribable[WorkersUpdate]):
    def __init__(self):
        super().__init__()
        self.inner: t.Dict[WorkerID, Worker] = {}
        self._futs: t.List[asyncio.Task[None]] = []

    def state(self) -> t.List[WorkerState]:
        return [worker.state() for worker in self.inner.values()]

    async def _subscribe_to_worker(self, worker: Worker):
        async for msg in worker.subscribe():
            if msg.msg == 'status_change':
                await self.message_subscribers(
                    WorkersUpdate(msg, self.state())
                )

    async def add(self, worker: Worker):
        self.inner[worker.id] = worker

        event = WorkerUpdate(worker.id, worker.status)
        await self.message_subscribers(WorkersUpdate(event, self.state()))
        self._futs.append(asyncio.create_task(self._subscribe_to_worker(worker)))

    def schedule_for_removal(self, worker_id: WorkerID, delay: float = 30.0):
        if worker_id not in self:
            return

        async def task():
            async with server.app.app_context():
                await asyncio.sleep(delay)
                await self.remove(worker_id)

        self._futs.append(asyncio.create_task(task()))

    async def remove(self, worker_id: WorkerID):
        try:
            worker = self.inner.pop(worker_id)
        except KeyError:
            return

        await worker.finalize()
        await self.message_subscribers(
            WorkersUpdate(None, self.state())
        )

    def __contains__(self, item: WorkerID) -> bool:
        return self.inner.__contains__(item)

    def __getitem__(self, item: WorkerID) -> Worker:
        return self.inner[item]

    def items(self) -> t.ItemsView[WorkerID, Worker]:
        return self.inner.items()

    def keys(self) -> t.KeysView[WorkerID]:
        return self.inner.keys()

    def values(self) -> t.ValuesView[Worker]:
        return self.inner.values()

    async def finalize(self):
        for fut in self._futs:
            fut.cancel()
        await asyncio.gather(*(worker.finalize() for worker in self.inner.values()))


class Job(Subscribable[JobMessage]):
    def __init__(self, id: JobID, plan: str):
        super().__init__()
        self.id: JobID = id
        self.plan: str = plan
        self.status: JobStatus = 'queued'
        # cached job state
        self.cache: t.Dict[str, t.Any] = {}
        # and log messages
        self.logs: t.List[LogRecord] = []

    async def set_status(self, status: JobStatus):
        if self.status == status:
            return
        self.status = status
        await self.message_subscribers(
            JobStatusChange(status, self.id)
        )

    async def cancel(self):
        await self.set_status('stopping')

    def should_cancel(self) -> bool:
        return self.status == 'stopping'

    def state(self, full: bool = False) -> JobState:
        state = self.cache if full else {}
        return JobState.make_unchecked(self.id, self.status, {
            'dashboard': url_for('job_dashboard', job_id=self.id),
            'cancel': url_for('cancel_job', job_id=self.id), 
            'logs': url_for('job_logs', job_id=self.id),
        }, state=state)

    async def handle_update(self, msg: WorkerMessage):
        if msg.msg == 'job_update':
            if self.status in ('queued', 'starting'):
                await self.set_status('running')

            self.cache.update(msg.state)
            await self.message_subscribers(
                JobUpdate.make_unchecked(msg.state, msg.job_id)
            )
        elif msg.msg == 'log':
            record = msg.into_record(len(self.logs))
            self.logs.append(record)
            await self.message_subscribers(
                LogUpdate.make_unchecked([record])
            )
        elif msg.msg == 'job_result':
            self.status = 'stopped'
            await self.message_subscribers(
                JobStopped(msg.result, msg.error)
            )


class Jobs(Subscribable[JobsUpdate]):
    def __init__(self):
        super().__init__()
        self.inner: t.Dict[JobID, Job] = {}
        self._futs: t.List[asyncio.Task[None]] = []

    def state(self) -> t.List[JobState]:
        return [job.state() for job in self.inner.values()]

    async def _subscribe_to_job(self, job: Job):
        async for msg in job.subscribe():
            if msg.msg in ('status_change', 'job_stopped'):
                await self.message_subscribers(
                    JobsUpdate.make_unchecked(msg, self.state())
                )

    async def add(self, job: Job):
        self.inner[job.id] = job
        self._futs.append(asyncio.create_task(self._subscribe_to_job(job)))
        server.job_queue.append(job)

        event = JobStatusChange(job.status, job.id)
        await self.message_subscribers(
            JobsUpdate.make_unchecked(event, self.state())
        )

    def __contains__(self, item: JobID) -> bool:
        return self.inner.__contains__(item)

    def __getitem__(self, item: JobID) -> Job:
        return self.inner[item]

    def items(self) -> t.ItemsView[JobID, Job]:
        return self.inner.items()

    def keys(self) -> t.KeysView[JobID]:
        return self.inner.keys()

    def values(self) -> t.ValuesView[Job]:
        return self.inner.values()

    async def finalize(self):
        for fut in self._futs:
            fut.cancel()


_ID_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


class Server:
    def __init__(self):
        self.workers: Workers = Workers()
        self.jobs: Jobs = Jobs()
        self.job_queue: deque[Job] = deque()

        from .slurm import SlurmManager

        self.slurm_manager: SlurmManager = SlurmManager()

        self.app: Quart = Quart(
            __name__,
            static_url_path="/static",
            static_folder="dist",
        )
        self.app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 5

        @self.app.after_serving
        async def shutdown():
            await asyncio.gather(self.jobs.finalize(), self.workers.finalize(), self.slurm_manager.finalize())

    def run(self, plan: ReconsPlan):
        import json
        self.plan: str = json.dumps(plan.into_data())
        logging.basicConfig(level=logging.INFO)
        multiprocessing.set_start_method('spawn', True)

        self.app.run(host='0.0.0.0', port=5050, debug=True)

    def get_worker_url(self, worker_id: WorkerID) -> str:
        url = url_for('worker_update', worker_id=worker_id, _external=True)
        return url

    def make_workerid(self) -> WorkerID:
        while True:
            id = "".join(_ID_CHARS[random.randrange(0, len(_ID_CHARS))] for _ in range(10))
            if id not in self.workers:
                return id

    def make_jobid(self) -> JobID:
        while True:
            id = "".join(_ID_CHARS[random.randrange(0, len(_ID_CHARS))] for _ in range(10))
            if id not in self.jobs:
                return id


server: Server = Server()

from . import routes  # noqa: E402, F401