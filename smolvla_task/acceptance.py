"""Development model-selection gate, separate from final population testing."""


def stage6_admission(metrics):
    attempts = metrics["attempts"]
    successes = metrics["successes"]
    if not isinstance(attempts, int) or not isinstance(successes, int) or not 0 <= successes <= attempts:
        raise ValueError("Invalid closed-loop trial counts")
    # Engineering development target. Five trials do not establish a
    # population80% success rate, and this is not a mandatory PPT threshold.
    passed = attempts == 5 and successes >= 4
    return dict(passed=passed, attempts=attempts, full_task_successes=successes,
                engineering_development_gate="At least4 complete successes on5 fixed validation positions",
                population_success_rate_not_established=True,
                next_stage="freeze_then_step8" if passed else "step6_diagnosis")

