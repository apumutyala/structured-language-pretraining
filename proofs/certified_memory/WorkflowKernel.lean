import SoundMemory

/- An n=3 executable instantiation. Only the host supplies actions, authenticated
   observations and protected policy; an untrusted proposer supplies an 8-bit
   candidate support mask. Authentication and external execution are assumptions. -/
namespace WorkflowKernel

abbrev ExactMask := Fin 256
abbrev PriorMask := { mask : ExactMask // mask.val ≠ 0 }
abbrev Grants := Fin 8
abbrev Index := Fin 3

def bit (world index : Nat) : Bool := decide (world / 2 ^ index % 2 = 1)

def popcount (mask : Nat) : Nat :=
  (List.range 3).foldl (fun count i => count + if bit mask i then 1 else 0) 0

def maskMemory (mask : ExactMask) : CertifiedMemory.Memory :=
  (List.range 8).filter (fun world => bit mask.val world)

def supportMask (memory : CertifiedMemory.Memory) : Nat :=
  (List.range 8).foldl
    (fun mask world => mask + if world ∈ memory then 2 ^ world else 0) 0

inductive Action where
  | wait
  | submit (index : Index)
  deriving DecidableEq, Repr

inductive Observation where
  | timeout
  | untrusted
  | receipt (index : Index) (success : Bool)
  | parity (mask : Grants) (rhs : Bool)
  | count (mask : Grants) (value : Fin (popcount mask.val + 1))

def observationPredicate : Observation → Nat → Bool
  | .timeout, _ => true
  | .untrusted, _ => true
  | .receipt index success, world => bit world index.val == success
  | .parity mask rhs, world =>
      (popcount (world &&& mask.val) % 2 == 1) == rhs
  | .count mask value, world => popcount (world &&& mask.val) == value.val

def step : Action → Nat → CertifiedMemory.Memory
  | .wait, world => [world]
  | .submit index, world => [world, world ||| 2 ^ index.val]

structure State where
  memory : CertifiedMemory.Memory
  grants : Grants
  budget : Nat
  deriving DecidableEq

structure Request where
  prior : PriorMask
  grants : Grants
  budget : Nat
  action : Action
  observation : Observation
  candidate : ExactMask

def initialState (request : Request) : State :=
  ⟨maskMemory request.prior.val, request.grants, request.budget⟩

/-- This predicate concerns one possible world, including protected policy. -/
def actionPredicate (grants : Grants) (budget : Nat) : Action → Nat → Bool
  | .wait, _ => true
  | .submit index, world =>
      decide (budget > 0) && bit grants.val index.val && !bit world index.val

/-- The explicit nonempty guard prevents vacuous robust approval. -/
def legal (state : State) (action : Action) : Bool :=
  if state.memory = [] then false
  else CertifiedMemory.robustAction state.memory
    (actionPredicate state.grants state.budget action)

def chargedBudget (budget : Nat) : Action → Nat
  | .wait => budget
  | .submit _ => budget - 1

structure Result where
  state : State
  accepted : Bool
  contradiction : Bool
  actionPermitted : Bool
  deriving DecidableEq

/-- Illegal actions leave state unchanged. For legal events the exact generic
    checker handles conditioning, contradictory fallback and candidate inclusion. -/
def run (request : Request) : Result :=
  let prior := initialState request
  if legal prior request.action then
    let post := CertifiedMemory.unobservedPost (step request.action) prior.memory
    let conditioned := CertifiedMemory.condition post (observationPredicate request.observation)
    { state := {
        memory := CertifiedMemory.checkedUpdate prior.memory (step request.action)
          (observationPredicate request.observation) (maskMemory request.candidate)
        grants := prior.grants
        budget := chargedBudget prior.budget request.action }
      accepted := CertifiedMemory.candidateOK conditioned (maskMemory request.candidate)
      contradiction := decide (conditioned = [])
      actionPermitted := true }
  else
    { state := prior, accepted := false, contradiction := false, actionPermitted := false }

def legalSubmitMask (state : State) : Nat :=
  (if legal state (.submit ⟨0, by decide⟩) then 1 else 0) +
  (if legal state (.submit ⟨1, by decide⟩) then 2 else 0) +
  (if legal state (.submit ⟨2, by decide⟩) then 4 else 0)

def groundedObservation (mask : ExactMask) (observation : Observation) : Bool :=
  CertifiedMemory.groundedAssertion (maskMemory mask) (observationPredicate observation)

theorem mask_world_bounded (mask : ExactMask) (world : Nat)
    (member : world ∈ maskMemory mask) : world < 8 := by
  have enumerated := (List.mem_filter.mp member).1
  change world ∈ [0, 1, 2, 3, 4, 5, 6, 7] at enumerated
  simp only [List.mem_cons, List.not_mem_nil, or_false] at enumerated
  rcases enumerated with h | h | h | h | h | h | h | h <;> subst world <;> decide

private theorem submit_bounded : ∀ world : Fin 8, ∀ index : Index,
    world.val ||| 2 ^ index.val < 8 := by decide

theorem step_world_bounded (action : Action) (world successor : Nat)
    (bounded : world < 8) (member : successor ∈ step action world) : successor < 8 := by
  cases action with
  | wait =>
    have eq : successor = world := by simpa [step] using member
    simpa [eq] using bounded
  | submit index =>
    have alternatives : successor = world ∨ successor = world ||| 2 ^ index.val := by
      simpa [step] using member
    rcases alternatives with equal | equal
    · simpa [equal] using bounded
    · rw [equal]
      exact submit_bounded ⟨world, bounded⟩ index

private theorem member_append {world : Nat} {left right : List Nat}
    (member : world ∈ left ++ right) : world ∈ left ∨ world ∈ right := by
  induction left with
  | nil => exact Or.inr member
  | cons head tail ih =>
    rcases List.mem_cons.mp member with first | rest
    · exact Or.inl (List.mem_cons.mpr (Or.inl first))
    · rcases ih rest with inTail | inRight
      · exact Or.inl (List.mem_cons.mpr (Or.inr inTail))
      · exact Or.inr inRight

private theorem post_world_bounded (action : Action) (prior : CertifiedMemory.Memory)
    (bounded : ∀ world, world ∈ prior → world < 8) :
    ∀ successor, successor ∈ CertifiedMemory.unobservedPost (step action) prior → successor < 8 := by
  induction prior with
  | nil => intro successor member; exact False.elim (List.not_mem_nil successor member)
  | cons head tail ih =>
    intro successor member
    have either := member_append member
    rcases either with first | rest
    · exact step_world_bounded action head successor
        (bounded head (List.mem_cons.mpr (Or.inl rfl))) first
    · exact ih (fun w h => bounded w (List.mem_cons.mpr (Or.inr h))) successor rest

/-- Every executable result denotes only the eight declared worlds. -/
theorem result_world_bounded (request : Request) (world : Nat)
    (member : world ∈ (run request).state.memory) : world < 8 := by
  have priorBound : ∀ w, w ∈ (initialState request).memory → w < 8 :=
    mask_world_bounded request.prior.val
  have postBound := post_world_bounded request.action (initialState request).memory priorBound
  by_cases allowed : legal (initialState request) request.action = true
  · simp only [run, allowed, if_true] at member
    simp only [CertifiedMemory.checkedUpdate] at member
    split at member
    · exact postBound world member
    · split at member
      · exact mask_world_bounded request.candidate world member
      · exact postBound world (List.mem_filter.mp member).1
  · simpa only [run, allowed, if_false] using priorBound world (by
      simpa only [run, allowed, if_false] using member)

theorem illegal_no_mutation (request : Request)
    (illegal : legal (initialState request) request.action = false) :
    (run request).state = initialState request ∧
      (run request).accepted = false ∧ (run request).actionPermitted = false := by
  simp [run, illegal]

theorem grants_preserved (request : Request) :
    (run request).state.grants = request.grants := by
  by_cases allowed : legal (initialState request) request.action = true
  · simp only [run, allowed, if_true]
    rfl
  · simp only [run, allowed, if_false]
    rfl

theorem legal_budget (request : Request)
    (allowed : legal (initialState request) request.action = true) :
    (run request).state.budget = chargedBudget request.budget request.action := by
  simp only [run, allowed, if_true]
  rfl

theorem legal_submit_charges (request : Request) (index : Index)
    (action : request.action = .submit index)
    (allowed : legal (initialState request) request.action = true) :
    (run request).state.budget = request.budget - 1 := by
  rw [legal_budget request allowed, action]
  rfl

theorem legal_wait_budget (request : Request)
    (action : request.action = .wait)
    (allowed : legal (initialState request) request.action = true) :
    (run request).state.budget = request.budget := by
  rw [legal_budget request allowed, action]
  rfl

/-- The actual world must be in the prior, the successor must be a declared
    action outcome, and the observation must be truthful OR contradictory with
    every possible post world. A false but consistent receipt is excluded. -/
theorem successor_contained (request : Request) (actual successor : Nat)
    (allowed : legal (initialState request) request.action = true)
    (contains : actual ∈ (initialState request).memory)
    (transition : successor ∈ step request.action actual)
    (observationSemantics : observationPredicate request.observation successor = true ∨
      CertifiedMemory.condition
        (CertifiedMemory.unobservedPost (step request.action) (initialState request).memory)
        (observationPredicate request.observation) = []) :
    successor ∈ (run request).state.memory := by
  have proof := CertifiedMemory.checked_update_survives
    (candidate := maskMemory request.candidate) contains transition observationSemantics
  simpa [run, allowed] using proof

theorem successor_nonempty (request : Request) (actual successor : Nat)
    (allowed : legal (initialState request) request.action = true)
    (contains : actual ∈ (initialState request).memory)
    (transition : successor ∈ step request.action actual)
    (observationSemantics : observationPredicate request.observation successor = true ∨
      CertifiedMemory.condition
        (CertifiedMemory.unobservedPost (step request.action) (initialState request).memory)
        (observationPredicate request.observation) = []) :
    (run request).state.memory ≠ [] := by
  intro empty
  have member := successor_contained request actual successor allowed contains transition observationSemantics
  rw [empty] at member
  exact List.not_mem_nil successor member

theorem approved_action_actual (state : State) (action : Action) (actual : Nat)
    (contains : actual ∈ state.memory) (approved : legal state action = true) :
    actionPredicate state.grants state.budget action actual = true := by
  unfold legal at approved
  split at approved
  · contradiction
  · exact CertifiedMemory.robust_action_actual contains approved

theorem checked_action_actual (request : Request) (next : Action) (actual successor : Nat)
    (allowed : legal (initialState request) request.action = true)
    (contains : actual ∈ (initialState request).memory)
    (transition : successor ∈ step request.action actual)
    (observationSemantics : observationPredicate request.observation successor = true ∨
      CertifiedMemory.condition
        (CertifiedMemory.unobservedPost (step request.action) (initialState request).memory)
        (observationPredicate request.observation) = [])
    (approved : legal (run request).state next = true) :
    actionPredicate (run request).state.grants (run request).state.budget next successor = true := by
  exact approved_action_actual (run request).state next successor
    (successor_contained request actual successor allowed contains transition observationSemantics) approved

theorem contradiction_fallback (request : Request)
    (allowed : legal (initialState request) request.action = true)
    (contradiction : CertifiedMemory.condition
      (CertifiedMemory.unobservedPost (step request.action) (initialState request).memory)
      (observationPredicate request.observation) = []) :
    (run request).state.memory =
      CertifiedMemory.unobservedPost (step request.action) (initialState request).memory ∧
    (run request).accepted = false ∧ (run request).contradiction = true := by
  simp [run, allowed, CertifiedMemory.checkedUpdate, contradiction,
    CertifiedMemory.candidateOK]

theorem incomplete_candidate_fallback (request : Request)
    (allowed : legal (initialState request) request.action = true)
    (nonempty : CertifiedMemory.condition
      (CertifiedMemory.unobservedPost (step request.action) (initialState request).memory)
      (observationPredicate request.observation) ≠ [])
    (rejected : CertifiedMemory.candidateOK
      (CertifiedMemory.condition
        (CertifiedMemory.unobservedPost (step request.action) (initialState request).memory)
        (observationPredicate request.observation)) (maskMemory request.candidate) = false) :
    (run request).state.memory = CertifiedMemory.condition
      (CertifiedMemory.unobservedPost (step request.action) (initialState request).memory)
      (observationPredicate request.observation) ∧ (run request).accepted = false := by
  simp [run, allowed, CertifiedMemory.checkedUpdate, nonempty, rejected]

theorem sound_candidate_accepted (request : Request)
    (allowed : legal (initialState request) request.action = true)
    (nonempty : CertifiedMemory.condition
      (CertifiedMemory.unobservedPost (step request.action) (initialState request).memory)
      (observationPredicate request.observation) ≠ [])
    (covers : ∀ w, w ∈ CertifiedMemory.condition
      (CertifiedMemory.unobservedPost (step request.action) (initialState request).memory)
      (observationPredicate request.observation) → w ∈ maskMemory request.candidate) :
    (run request).accepted = true ∧ (run request).state.memory = maskMemory request.candidate := by
  have accepted : CertifiedMemory.candidateOK
      (CertifiedMemory.condition
        (CertifiedMemory.unobservedPost (step request.action) (initialState request).memory)
        (observationPredicate request.observation)) (maskMemory request.candidate) = true := by
    simp only [CertifiedMemory.candidateOK, if_neg nonempty]
    exact List.all_eq_true.mpr (fun w member => decide_eq_true (covers w member))
  simp [run, allowed, CertifiedMemory.checkedUpdate, nonempty, accepted]

theorem grounded_observation_actual (mask : ExactMask) (observation : Observation)
    (approved : groundedObservation mask observation = true) :
    maskMemory mask ≠ [] ∧ ∀ actual, actual ∈ maskMemory mask →
      observationPredicate observation actual = true := by
  exact CertifiedMemory.grounded_assertion_sound approved

example : groundedObservation ⟨0, by decide⟩ .timeout = false := by decide
example : legal ⟨[], ⟨7, by decide⟩, 3⟩ .wait = false := by decide

#print axioms illegal_no_mutation
#print axioms result_world_bounded
#print axioms grants_preserved
#print axioms legal_budget
#print axioms legal_submit_charges
#print axioms legal_wait_budget
#print axioms successor_contained
#print axioms successor_nonempty
#print axioms approved_action_actual
#print axioms checked_action_actual
#print axioms contradiction_fallback
#print axioms incomplete_candidate_fallback
#print axioms sound_candidate_accepted
#print axioms grounded_observation_actual

end WorkflowKernel
