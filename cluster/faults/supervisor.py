"""Deadline supervision of one exclusive delegated cgroup tree; apply is paused."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from cluster.faults.plan import PAUSED


class _OwnedTrainingGroup:
    """Gate the launcher before exec; cgroup population tracks every descendant."""

    def __init__(self, argv, *, env, popen, scope, clock=time.monotonic, sleep=time.sleep):
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(a, str) or not a for a in argv)
        ):
            raise ValueError("explicit argv vector required")
        if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
            raise ValueError("single owned-child reaper with default SIGCHLD required")
        if scope.populated():
            raise ValueError("exclusive cgroup must be empty before launch")
        self.scope, self.clock, self.sleep = scope, clock, sleep
        read, write = os.pipe()
        try:
            gate = str(Path(__file__).with_name("scope.py"))
            self.child = popen(
                [sys.executable, "-I", "-S", gate, "--gate", str(read), "--", *argv],
                env=env,
                start_new_session=True,
                shell=False,
                pass_fds=(read,),
            )
            # Do not poll/reap this blocked direct child before attaching its PID:
            # an unreaped child PID cannot be recycled under default SIGCHLD.
            scope.attach(self.child.pid)
            os.write(write, b"1")
        finally:
            os.close(read)
            os.close(write)  # EOF on failure forbids training exec/fork.

    def wait(self, timeout):
        deadline = self.clock() + timeout
        self.scope.wait_empty(timeout, clock=self.clock, sleep=self.sleep)
        # A reaped leader alone never constitutes group completion. Only the
        # recursive cgroup.events populated=0 observation permits direct-child reap.
        return self.child.wait(timeout=max(0, deadline - self.clock()))

    def terminate_group(self):
        self.scope.terminate()  # pidfd TERM of currently verified owned members.

    def kill_group(self):
        self.scope.kill()  # cgroup.kill covers subtree and concurrent forks.


class CampaignSupervisor:
    def apply(self, *args, **kwargs):
        raise PermissionError(PAUSED)

    def rehearse(
        self,
        group,
        *,
        started_at,
        clock,
        read_events=list,
        emit=None,
        wait_until=None,
        terminate_grace_seconds=5,
    ):
        from scaling.goodput import account_events, worker_state

        if type(terminate_grace_seconds) is not int or terminate_grace_seconds <= 0:
            raise ValueError("positive bounded termination grace required")
        evidence = []
        result = {"simulation": True, "status": "failed", "valid": False, "evidence": evidence}
        exited = False
        deadline = started_at + 7200
        try:
            remaining = deadline - clock()
            if remaining <= 0:
                raise TimeoutError("campaign deadline reached")
            code = group.wait(timeout=remaining)
            exited, exit_at = True, clock()
            evidence.append(
                {"kind": "scope_empty_and_launcher_reaped", "at": exit_at, "code": code}
            )
            if exit_at > deadline:
                raise TimeoutError("whole worker scope exit not confirmed within work budget")
            events = list(read_events())
            if (
                not events
                or events[0].get("at") != started_at
                or events[-1].get("at", float("inf")) > exit_at
                or code != 0
            ):
                raise ValueError("clean whole-scope exit and observed worker journal required")
            state = worker_state(events)
            if emit is None or wait_until is None:
                raise ValueError("surviving coordinator journal sink/wait required")
            if clock() < deadline:
                wait_until(deadline)
            observed = clock()
            if observed < deadline:
                raise ValueError("coordinator observed before the planned window boundary")
            # Timestamp is the actual observation, never a fabricated exact deadline.
            terminal = {k: events[0][k] for k in ("schema", "run_id", "clock_id", "rank")}
            terminal.update(
                kind="run_end",
                at=observed,
                checkpoint_id=state["final_checkpoint_id"],
                planned_window_end=deadline,
                workers_exited_at=exit_at,
                observation_lateness_seconds=observed - deadline,
            )
            emit(terminal)
            events.append(terminal)
            result.update(
                status="complete",
                valid=True,
                accounting=account_events(events),
                planned_wall_time=7200,
                observation_lateness_seconds=observed - deadline,
            )
            return result
        except (TimeoutError, subprocess.TimeoutExpired) as exc:
            result.update(status="deadline_exceeded", error=str(exc))
        except Exception as exc:  # noqa: BLE001 - retain evidence, never infer success
            result["error"] = str(exc)
        if exited:
            return result  # Empty scope/leader already reaped: never signal an old identifier.
        for operation in ("terminate_group", "kill_group"):
            try:
                getattr(group, operation)()
                evidence.append({"kind": operation, "at": clock()})
                result["exit_code"] = group.wait(timeout=terminate_grace_seconds)
                break
            except (TimeoutError, subprocess.TimeoutExpired):
                continue
            except Exception as exc:  # noqa: BLE001 - ownership failure stops unsafe operations
                evidence.append({"kind": operation, "at": clock(), "error": str(exc)})
        else:
            result["cleanup_error"] = (
                "owned subtree empty/reap not confirmed after bounded escalation"
            )
        return result

    def rehearse_launch(
        self,
        argv,
        *,
        run_id,
        clock_id,
        clock,
        emit,
        scope,
        read_events=list,
        env=None,
        popen,
        wait_until=None,
        sleep=time.sleep,
    ):
        """Only explicitly supplied CPU Popen/scope shims are supported today."""
        environment = dict(os.environ if env is None else env)
        origin = {
            "schema": 1,
            "rank": 0,
            "kind": "run_start",
            "run_id": run_id,
            "clock_id": clock_id,
            "world_size": int(environment.get("WORLD_SIZE", "1")),
            "at": clock(),
        }
        emit(origin)
        environment["CRUCIBLE_CAMPAIGN_ORIGIN_JSON"] = json.dumps(origin, allow_nan=False)
        group = _OwnedTrainingGroup(
            argv, env=environment, popen=popen, scope=scope, clock=clock, sleep=sleep
        )
        return self.rehearse(
            group,
            started_at=origin["at"],
            clock=clock,
            read_events=read_events,
            emit=emit,
            wait_until=wait_until,
        )
