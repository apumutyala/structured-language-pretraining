import Init

namespace CertifiedMemory

abbrev World := Nat
abbrev Memory := List World

/-- The declared transition may be nondeterministic; this list retains all branches. -/
def unobservedPost (step : World → Memory) : Memory → Memory
  | [] => []
  | w :: ws => step w ++ unobservedPost step ws

def condition (post : Memory) (observation : World → Bool) : Memory :=
  post.filter observation

/-- Computed inclusion, guarded against vacuous acceptance. -/
def candidateOK (conditioned candidate : Memory) : Bool :=
  if conditioned = [] then false
  else conditioned.all (fun w => decide (w ∈ candidate))

/-- Contradictory observations discard the observation, never the prior post.
    Other rejections preserve every observation-conditioned possibility. -/
def checkedUpdate (prior : Memory) (step : World → Memory)
    (observation : World → Bool) (candidate : Memory) : Memory :=
  let post := unobservedPost step prior
  let conditioned := condition post observation
  if conditioned = [] then post
  else if candidateOK conditioned candidate then candidate else conditioned

private theorem append_left {w : World} {left right : Memory}
    (member : w ∈ left) : w ∈ left ++ right := by
  induction left with
  | nil => exact False.elim (List.not_mem_nil w member)
  | cons head tail ih =>
    rcases List.mem_cons.mp member with h | h
    · exact List.mem_cons.mpr (Or.inl h)
    · exact List.mem_cons.mpr (Or.inr (ih h))

private theorem append_right {w : World} {left right : Memory}
    (member : w ∈ right) : w ∈ left ++ right := by
  induction left with
  | nil => exact member
  | cons head tail ih => exact List.mem_cons.mpr (Or.inr ih)

theorem post_covers {prior : Memory} {step : World → Memory} {actual successor : World}
    (priorContains : actual ∈ prior) (transition : successor ∈ step actual) :
    successor ∈ unobservedPost step prior := by
  induction prior with
  | nil => exact False.elim (List.not_mem_nil actual priorContains)
  | cons head tail ih =>
    rcases List.mem_cons.mp priorContains with h | h
    · subst head
      exact append_left transition
    · exact append_right (ih h)

theorem candidateOK_sound {conditioned candidate : Memory}
    (accepted : candidateOK conditioned candidate = true) :
    conditioned ≠ [] ∧ ∀ w, w ∈ conditioned → w ∈ candidate := by
  unfold candidateOK at accepted
  split at accepted
  · contradiction
  · rename_i nonempty
    constructor
    · exact nonempty
    · intro w member
      exact of_decide_eq_true ((List.all_eq_true.mp accepted) w member)

/-- A truthful observation keeps the actual successor. If conditioning is
    contradictory, its fallback retains every declared successor instead.
    A false but consistent observation is deliberately outside this theorem. -/
theorem checked_update_survives {prior : Memory} {step : World → Memory}
    {observation : World → Bool} {candidate : Memory} {actual successor : World}
    (priorContains : actual ∈ prior) (transition : successor ∈ step actual)
    (observationSemantics : observation successor = true ∨
      condition (unobservedPost step prior) observation = []) :
    successor ∈ checkedUpdate prior step observation candidate := by
  have hp := post_covers priorContains transition
  by_cases empty : condition (unobservedPost step prior) observation = []
  · simpa [checkedUpdate, empty] using hp
  · have truthful : observation successor = true := observationSemantics.resolve_right empty
    have hc : successor ∈ condition (unobservedPost step prior) observation :=
      List.mem_filter.mpr ⟨hp, truthful⟩
    by_cases accepted : candidateOK (condition (unobservedPost step prior) observation) candidate = true
    · have member := (candidateOK_sound accepted).2 successor hc
      simpa [checkedUpdate, empty, accepted] using member
    · simpa [checkedUpdate, empty, accepted] using hc

theorem checked_update_nonempty {prior : Memory} {step : World → Memory}
    {observation : World → Bool} {candidate : Memory} {actual successor : World}
    (priorContains : actual ∈ prior) (transition : successor ∈ step actual)
    (observationSemantics : observation successor = true ∨
      condition (unobservedPost step prior) observation = []) :
    checkedUpdate prior step observation candidate ≠ [] := by
  intro empty
  have member := checked_update_survives (candidate := candidate) priorContains transition observationSemantics
  rw [empty] at member
  exact List.not_mem_nil successor member

/-- These are checks on every retained world, not a claim about arbitrary prose. -/
def robustAction (memory : Memory) (actionPredicate : World → Bool) : Bool :=
  memory.all actionPredicate

theorem robust_action_actual {memory : Memory} {actual : World}
    {actionPredicate : World → Bool} (contains : actual ∈ memory)
    (approved : robustAction memory actionPredicate = true) :
    actionPredicate actual = true := by
  exact (List.all_eq_true.mp approved) actual contains

theorem checked_action_actual {prior : Memory} {step : World → Memory}
    {observation actionPredicate : World → Bool} {candidate : Memory} {actual successor : World}
    (priorContains : actual ∈ prior) (transition : successor ∈ step actual)
    (observationSemantics : observation successor = true ∨
      condition (unobservedPost step prior) observation = [])
    (approved : robustAction (checkedUpdate prior step observation candidate) actionPredicate = true) :
    actionPredicate successor = true := by
  exact robust_action_actual (checked_update_survives priorContains transition observationSemantics) approved

def groundedAssertion (memory : Memory) (predicate : World → Bool) : Bool :=
  if memory = [] then false else memory.all predicate

theorem grounded_assertion_sound {memory : Memory} {predicate : World → Bool}
    (approved : groundedAssertion memory predicate = true) :
    memory ≠ [] ∧ ∀ actual, actual ∈ memory → predicate actual = true := by
  unfold groundedAssertion at approved
  split at approved
  · contradiction
  · exact ⟨by assumption, List.all_eq_true.mp approved⟩

/-- A tiny count model: a timeout may leave the count unchanged or increment it. -/
def timeoutStep (w : World) : Memory := [w, w + 1]

def zeroCount (w : World) : Bool := decide (w = 0)

def atMostOne (w : World) : Bool := decide (w ≤ 1)

-- Timeout gives no success/failure evidence. A learned singleton would forget
-- one branch and is rejected by the executable inclusion checker.
example : checkedUpdate [0] timeoutStep (fun _ => true) [0] = [0, 1] := by decide
example : candidateOK [0, 1] [0] = false := by decide
-- A declared count observation retains exactly the count-consistent branch.
example : checkedUpdate [0] timeoutStep zeroCount [0] = [0] := by decide
-- A contradictory count is discarded, preserving the unobserved post.
example : checkedUpdate [0] timeoutStep (fun w => decide (w = 7)) [] = [0, 1] := by decide
example : groundedAssertion [] atMostOne = false := by decide
example : groundedAssertion [0, 1] atMostOne = true := by decide
example : robustAction [0, 1] zeroCount = false := by decide

#eval checkedUpdate [0] timeoutStep (fun _ => true) [0]
#eval checkedUpdate [0] timeoutStep zeroCount [0]
#eval checkedUpdate [0] timeoutStep (fun w => decide (w = 7)) []
#print axioms post_covers
#print axioms candidateOK_sound
#print axioms checked_update_survives
#print axioms checked_update_nonempty
#print axioms robust_action_actual
#print axioms checked_action_actual
#print axioms grounded_assertion_sound

end CertifiedMemory
