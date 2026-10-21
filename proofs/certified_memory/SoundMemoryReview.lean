import SoundMemory

namespace CertifiedMemoryReview

open CertifiedMemory

-- Rejected stale summaries retain the possible successful successor.
theorem stale_prior_candidate_is_rejected :
    candidateOK [0, 1] [0] = false ∧
    checkedUpdate [0] timeoutStep (fun _ => true) [0] = [0, 1] := by decide

-- A contradictory observation cannot select even a generously broad candidate;
-- the result is the unobserved post-state, rather than the stale prior.
theorem contradiction_retains_unobserved_post :
    checkedUpdate [0] timeoutStep (fun w => decide (w = 7)) [7] = [0, 1] ∧
    1 ∈ checkedUpdate [0] timeoutStep (fun w => decide (w = 7)) [7] := by decide

-- This closed counterexample demonstrates why truthful/current evidence is an
-- explicit theorem hypothesis: a false but consistent observation can omit 1.
theorem false_consistent_observation_can_drop_actual :
    1 ∈ timeoutStep 0 ∧
    checkedUpdate [0] timeoutStep zeroCount [0] = [0] ∧
    ¬ 1 ∈ checkedUpdate [0] timeoutStep zeroCount [0] := by decide

-- The theorem covers declared successors only; no relationship to a separate
-- physical environment or Python transition is established by these definitions.
theorem undeclared_successor_is_not_protected :
    ¬ 1 ∈ checkedUpdate [0] (fun _ => [0]) (fun _ => true) [0] := by decide

-- Nonempty prior alone is insufficient if the declared transition has no branch.
theorem nonempty_prior_does_not_ensure_successor :
    checkedUpdate [0] (fun _ => []) (fun _ => true) [0] = [] := by decide

-- robust_action_actual needs its actual-membership premise: all on an empty
-- list is vacuously true, whereas the public assertion guard rejects emptiness.
theorem empty_action_check_has_no_unconditional_safety_claim :
    robustAction [] (fun _ => false) = true ∧
    groundedAssertion [] (fun _ => false) = false := by decide

#print axioms stale_prior_candidate_is_rejected
#print axioms contradiction_retains_unobserved_post
#print axioms false_consistent_observation_can_drop_actual
#print axioms undeclared_successor_is_not_protected
#print axioms nonempty_prior_does_not_ensure_successor
#print axioms empty_action_check_has_no_unconditional_safety_claim

end CertifiedMemoryReview
