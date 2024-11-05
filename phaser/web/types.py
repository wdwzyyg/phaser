import logging
import typing as t

import pane
from pane.annotations import Tagged

from phaser.types import Cancelled
from .util import ReconsStateConverter


JobID: t.TypeAlias = str
WorkerID: t.TypeAlias = str

WorkerStatus = t.Literal['queued', 'starting', 'idle', 'running', 'stopping', 'stopped', 'unknown']
"""
Enum indicating the state of the connection, as described by either the server or client.

- 'queued': Worker is queued/scheduled to start
- 'starting': Worker is actively being started
- 'idle': No job is running
- 'running': Job is running
- 'stopping': Worker is being stopped
- 'stopped': Worker has been stopped (normally or abnormally)

- 'unknown': Job hasn't talked in a while, unknown state
"""

JobStatus = t.Literal['queued', 'starting', 'running', 'stopping', 'stopped']
"""
Enum indicating the state of a reconstruction job
"""

Result = t.Literal['finished', 'errored', 'cancelled', 'interrupted']
"""
Enum indicating a job or worker result.

- 'finished': Finished normally
- 'errored': Encounted an error running
- 'cancelled': Cancelled/shutdown by the user (from server side)
- 'interrupted': Cancelled/shutdown by the client/OS
"""

# server -> client messages

class WorkerState(pane.PaneBase):
    worker_id: WorkerID
    status: WorkerStatus
    links: t.Dict[str, str] = pane.field(default_factory=dict)

class WorkerUpdate(pane.PaneBase):
    worker_id: WorkerID
    status: WorkerStatus

    msg: t.Literal['status_change'] = 'status_change'

class JobState(pane.PaneBase):
    job_id: JobID
    status: JobStatus
    links: t.Dict[str, str] = pane.field(default_factory=dict)
    state: t.Dict[str, t.Any] = pane.field(converter=ReconsStateConverter(), default_factory=dict)
    worker_id: t.Optional[WorkerID] = None

    def strip_state(self) -> t.Self:
        return type(self)(
            self.job_id, self.status, self.links
        )

class JobStatusChange(pane.PaneBase):
    status: JobStatus
    job_id: JobID

    msg: t.Literal['status_change'] = 'status_change'

class JobUpdate(pane.PaneBase):
    state: t.Dict[str, t.Any] = pane.field(converter=ReconsStateConverter())
    job_id: JobID

    msg: t.Literal['job_update'] = 'job_update'

class JobStopped(pane.PaneBase):
    result: Result
    error: t.Optional[str] = None

    msg: t.Literal['job_stopped'] = 'job_stopped'

JobMessage: t.TypeAlias = t.Annotated[t.Union[
    JobStatusChange, JobUpdate, JobStopped
], Tagged('msg')]

class DashboardConnected(pane.PaneBase):
    state: JobState
    msg: t.Literal['connected'] = 'connected'

DashboardMessage: t.TypeAlias = t.Annotated[t.Union[
    JobStatusChange, JobUpdate, JobStopped, DashboardConnected
], Tagged('msg')]

class WorkersUpdate(pane.PaneBase):
    event: t.Optional[WorkerUpdate]
    state: t.List[WorkerState]

    msg: t.Literal['workers_update'] = 'workers_update'

class JobsUpdate(pane.PaneBase):
    event: JobMessage
    state: t.List[JobState]

    msg: t.Literal['jobs_update'] = 'jobs_update'

class ManagerConnected(pane.PaneBase):
    workers: t.List[WorkerState]
    jobs: t.List[JobState]

    msg: t.Literal['connected'] = 'connected'

ManagerMessage: t.TypeAlias = t.Annotated[t.Union[
    JobsUpdate, WorkersUpdate, ManagerConnected
], Tagged('msg')]

# worker -> server messages

class PollMessage(pane.PaneBase):
    """Message polling the server for a job"""
    msg: t.Literal['poll'] = 'poll'

class UpdateMessage(pane.PaneBase):
    """Message containing some state update"""
    state: t.Dict[str, t.Any] = pane.field(converter=ReconsStateConverter())
    job_id: JobID
    msg: t.Literal['job_update'] = 'job_update'

class LogMessage(pane.PaneBase):
    """Message corresponding to a log entry"""
    job_id: WorkerID

    log: str

    logger_name: str
    log_level: int

    line_number: int
    func_name: t.Optional[str] = None
    stack_info: t.Optional[str] = None

    msg: t.Literal['log'] = 'log'

    @classmethod
    def from_logrecord(cls, job_id: JobID, record: logging.LogRecord) -> t.Self:
        return cls(
            log=record.getMessage(),
            job_id=job_id,
            logger_name=record.name,
            log_level=record.levelno,
            line_number=record.lineno,
            func_name=record.funcName,
            stack_info=record.stack_info,
        )

class JobResultMessage(pane.PaneBase):
    """Message indicating job has stopped (normally or abnormally)"""
    job_id: JobID
    result: Result
    error: t.Optional[str] = None

    msg: t.Literal['job_result'] = 'job_result'

class WorkerShutdownMessage(pane.PaneBase):
    """Message indicating a worker has been stopped"""
    result: Result
    error: t.Optional[str] = None
    detail: t.Optional[str] = None

    msg: t.Literal['shutdown'] = 'shutdown'


# server -> worker responses

class JobResponse(pane.PaneBase):
    """Response containing a job to run"""
    job_id: JobID
    plan: str
    msg: t.Literal['job'] = 'job'

class CancelResponse(pane.PaneBase):
    """Response from server indicating client should cancel job or shutdown"""
    shutdown: bool = False
    urgent: bool = False
    msg: t.Literal['cancel'] = 'cancel'

class OkResponse(pane.PaneBase):
    """Response from server indicating no action should be taken"""
    msg: t.Literal['ok'] = 'ok'

WorkerMessage: t.TypeAlias = t.Annotated[t.Union[
    PollMessage, UpdateMessage,
    LogMessage, JobResultMessage, WorkerShutdownMessage,
], Tagged('msg')]

ServerResponse: t.TypeAlias = t.Annotated[t.Union[
    JobResponse, CancelResponse, OkResponse
], Tagged('msg')]


class JobCancelled(Cancelled):
    def __init__(self, shutdown: bool = False, urgent: bool = False):
        self.shutdown: bool = shutdown
        self.urgent: bool = urgent