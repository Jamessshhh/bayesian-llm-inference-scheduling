from __future__ import annotations
from typing import List

from latent.policies.base import Policy
from latent.sim.request import Request


class Simulator:
    """Single worker, non-preemptive. Whenever the server frees up,
    the policy picks the next request from everything that has arrived."""

    def __init__(self, policy: Policy):
        self.policy = policy

    def run(self, requests: List[Request]) -> List[Request]:
        pending = sorted(requests, key=lambda r: (r.arrival_time, r.id))
        n = len(pending)
        i = 0
        now = 0.0
        queue: List[Request] = []
        done: List[Request] = []

        while i < n or queue:
            if not queue:                       # idle: jump to next arrival
                now = max(now, pending[i].arrival_time)
            while i < n and pending[i].arrival_time <= now:
                queue.append(pending[i])
                i += 1

            req = self.policy.select(queue, now)
            queue.remove(req)

            req.start_time = now
            now += req.service_time
            req.finish_time = now
            done.append(req)

        return done