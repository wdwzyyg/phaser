import dataclasses
import logging
import sys
import traceback
import typing as t

import backoff
import requests

import pane

from phaser.execute import execute_plan, ReconsPlan
from phaser.state import ReconsState, PartialReconsState

from .types import (
    PollMessage, UpdateMessage, LogMessage, JobResultMessage,
    WorkerShutdownMessage, WorkerMessage, ServerResponse
)
from .types import JobID, JobCancelled


class LogHandler(logging.Handler):
    def __init__(self, send_message: t.Callable[[WorkerMessage], ServerResponse]):
        self.send_message = send_message
        self.job_id: t.Optional[JobID] = None

        super().__init__(logging.DEBUG)

    def emit(self, record: logging.LogRecord):
        if getattr(record, 'local', False):
            # local-only logging event
            return
        try:
            self.send_message(LogMessage.from_logrecord(self.job_id, record))
        except Exception:
            self.handleError(record)


def run_worker(url: str):
    def send_message(msg: WorkerMessage) -> ServerResponse:
        resp: requests.Response = requests.post(url, json=pane.into_data(msg))
        resp.raise_for_status()
        return pane.convert(resp.json(), ServerResponse)  # type: ignore

    # make inital connection to server
    # this has a relatively short backoff, so we can give up early
    @backoff.on_exception(backoff.fibo, requests.RequestException,
                          max_tries=10, max_time=30,
                          giveup=lambda e: isinstance(e, requests.HTTPError))  # giveup on 404, etc.
    def startup() -> ServerResponse:
        return send_message(PollMessage())

    # poll for a job from the server
    # for timeouts, we will eventualy fail and exit the loop
    # if we receive a response, however, we loop forever
    @backoff.on_predicate(backoff.fibo, lambda resp: resp.msg == 'ok',
                          max_value=10)
    @backoff.on_exception(backoff.fibo, requests.RequestException,
                          max_value=10, max_tries=60)
    @backoff.on_exception(backoff.fibo, requests.Timeout,
                          max_tries=10, max_time=60)
    def poll() -> ServerResponse:
        return send_message(PollMessage())

    # send job result
    @backoff.on_exception(backoff.fibo, requests.RequestException,
                          max_tries=10, max_time=30)
    def send_result(msg: JobResultMessage) -> ServerResponse:
        return send_message(msg)

    # and observe state updates
    def observer(job_id: JobID):
        def observe(state: t.Union[ReconsState, PartialReconsState]):
            try:
                resp = send_message(UpdateMessage.make_unchecked(
                    {k: v for (k, v) in dataclasses.asdict(state.to_numpy()).items() if v is not None},
                    job_id
                ))
            except (requests.RequestException, pane.ConvertError):
                logging.error("Failed to update server", exc_info=sys.exc_info(), extra={'local': True})
            else:
                if resp.msg == 'cancel':
                    raise JobCancelled(resp.shutdown, resp.urgent)

        return observe

    log_handler = LogHandler(send_message)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), log_handler])
    logger = logging.getLogger('worker')

    try:
        shutdown = False
        resp = startup()
        logger.info("Worker connected to server, response: %r", resp, extra={'local': True})

        while True:
            if resp.msg == 'ok':
                # poll for a new job
                resp = poll()
                logger.info("Worker polled server, response: %r", resp, extra={'local': True})

            if resp.msg == 'cancel':
                break

            assert resp.msg == 'job'

            try:
                # run job
                log_handler.job_id = resp.job_id

                plan = ReconsPlan.from_jsons(resp.plan)
                execute_plan(plan, [observer(resp.job_id)])

            except JobCancelled as e:
                logger.info("Job cancelled", extra={'local': True})
                msg = JobResultMessage(resp.job_id, 'cancelled')
                shutdown |= e.shutdown
            except KeyboardInterrupt:
                logger.info("Job interrupted", extra={'local': True})
                msg = JobResultMessage(resp.job_id, 'interrupted')
            except BaseException:
                logger.info("Job stopped due to error", exc_info=True, stack_info=True, extra={'local': True})
                s = traceback.format_exc()
                msg = JobResultMessage(resp.job_id, 'errored', s)
            else:
                logger.info("Job finished successfully", extra={'local': True})
                print("Job finished")
                msg = JobResultMessage(resp.job_id, 'finished')

            resp = send_result(msg)
            log_handler.job_id = None

            if shutdown:
                break

    except BaseException:
        # disconnect message
        logger.error("Worker shutting down due to error", exc_info=True, stack_info=True, extra={'local': True})
        s = traceback.format_exc()
        send_message(WorkerShutdownMessage('errored', error=s))

    finally:
        # disconnect message
        logger.info("Worker shutting down normally", extra={'local': True})
        send_message(WorkerShutdownMessage('finished'))