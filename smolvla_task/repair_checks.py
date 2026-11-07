"""Reproduce the actual interrupted command and verify inward rounding."""
import json
from types import SimpleNamespace

import numpy as np

from .config import OUTPUT
from .environment import PandaTask, float32_interior_limits
from .runtime import setup, write


def run():
    setup()
    source = OUTPUT/"rollouts/formal_20000_v1-test-base-normal/seed2026102301"
    rows = [json.loads(s) for s in (source/"commands.jsonl").read_text().splitlines()]
    last = rows[-1]
    with PandaTask() as env:
        native = env.native
        low, high = float32_interior_limits(native.limits)
        assert np.all(low.astype(np.float64)>=native.limits[:,0])
        assert np.all(high.astype(np.float64)<=native.limits[:,1])
        native.previous = np.asarray(last["issued"],np.float32).astype(np.float64)
        state = np.asarray(last["state_after"],np.float32)
        observer = native.observer
        native.observer = SimpleNamespace(state=lambda:state)
        actions = np.load(source/"chunk-046.npz")["actions"]
        raw = actions[len(rows)%25]
        # Original arithmetic, followed by the native layer's float32 cast.
        bounded=np.clip(raw.astype(np.float64),native.limits[:,0],native.limits[:,1])
        tracked=bounded.copy()
        tracked[:7]=np.clip(tracked[:7],state[:7]-.08,state[:7]+.08)
        increments=np.r_[np.full(7,.03),.06]
        broken=(native.previous+np.clip(tracked-native.previous,-increments,increments)).astype(np.float32).astype(np.float64)
        assert np.any(broken<native.limits[:,0]) or np.any(broken>native.limits[:,1]),"Must reproduce the original failure"
        fixed,_=env.limit_action(raw)
        assert np.all(fixed.astype(np.float64)>=native.limits[:,0]) and np.all(fixed.astype(np.float64)<=native.limits[:,1])
        assert np.max(np.abs(fixed.astype(float)-broken))<1e-6
        try:
            env.limit_action(np.full(8,np.nan))
            raise AssertionError("Nonfinite actions must still be rejected")
        except ValueError:
            pass
        # Test extreme predictions at all legal endpoints, independently of
        # whether the native position limit rounds upward or downward.
        for endpoint in [low,high]:
            native.previous=endpoint.astype(np.float64)
            state=np.r_[endpoint[:7],endpoint[7]].astype(np.float32)
            command,_=env.limit_action(endpoint.astype(np.float64)*100)
            assert np.all(command.astype(np.float64)>=native.limits[:,0])
            assert np.all(command.astype(np.float64)<=native.limits[:,1])
        native.observer=observer
        result=dict(passed=True,reproduced_command=len(rows)+1,original_invalid_target=broken,
                    corrected_target=fixed,max_correction_rad_or_aperture=float(np.max(abs(fixed.astype(float)-broken))),
                    inward_endpoints_verified=True,extreme_predictions_verified=True,nonfinite_still_rejected=True)
    write(OUTPUT/"evaluation-boundary-repair-verification.json",result)
    print(result,flush=True)

