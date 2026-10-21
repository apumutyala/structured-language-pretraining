import Init

/- A finite-list supported-record model, not a Python refinement theorem.
   Host evidence and rule truth, authentication, and deployment are assumptions.
   Nat identifiers abstract validated Python strings. See README.md. -/
namespace SupportedState

structure Claim where
  key : Nat
  value : Bool
  deriving DecidableEq, Repr

structure Rule where
  id : Nat
  premises : List Claim
  conclusion : Claim
  deriving DecidableEq, Repr

inductive Source where
  | evidence (id : Nat)
  | rule (id : Nat) (premises : List Nat)
  deriving DecidableEq, Repr

structure Record where
  id : Nat
  claim : Claim
  source : Source
  deriving DecidableEq, Repr

structure Fact where
  id : Nat
  claim : Claim
  deriving DecidableEq, Repr

def factRecord (fact : Fact) : Record := ⟨fact.id, fact.claim, .evidence fact.id⟩

structure State where
  session : Nat
  evidenceEpoch : Nat
  policyEpoch : Nat
  facts : List Fact
  rules : List Rule
  releaseKeys : List Nat
  records : List Record
  deriving DecidableEq, Repr

structure Certificate where
  session : Nat
  evidenceEpoch : Nat
  policyEpoch : Nat
  record : Record
  deriving DecidableEq, Repr

def certificate (state : State) (record : Record) : Certificate :=
  ⟨state.session, state.evidenceEpoch, state.policyEpoch, record⟩

def initial (session : Nat) (rules : List Rule) (keys : List Nat) : State :=
  ⟨session, 0, 0, [], rules, keys, []⟩

/-- Least closure of declared rules. With nonempty declared premises,
    support_has_evidence_root proves that even cyclic declarations need a root. -/
inductive Supported (facts : List Fact) (rules : List Rule) : Claim → Prop where
  | evidence (fact : Fact) : fact ∈ facts → Supported facts rules fact.claim
  | derive (rule : Rule) : rule ∈ rules →
      (∀ claim, claim ∈ rule.premises → Supported facts rules claim) →
      Supported facts rules rule.conclusion

def Invariant (state : State) : Prop :=
  ∀ record, record ∈ state.records → Supported state.facts state.rules record.claim

def Consistent (records : List Record) : Prop :=
  ∀ left, left ∈ records → ∀ right, right ∈ records →
    left.claim.key = right.claim.key → left.claim.value = right.claim.value

def EvidenceConsistent (facts : List Fact) : Prop :=
  ∀ left, left ∈ facts → ∀ right, right ∈ facts →
    left.claim.key = right.claim.key → left.claim.value = right.claim.value

/-- Success path of a host evidence mutation: even unchanged replacement resets
    all derived records and increments the evidence epoch. Host validates facts. -/
def revise (state : State) (facts : List Fact) : State :=
  { state with
    facts := facts
    evidenceEpoch := state.evidenceEpoch + 1
    records := facts.map factRecord }

def setPolicy (state : State) (keys : List Nat) : State :=
  { state with releaseKeys := keys, policyEpoch := state.policyEpoch + 1 }

def derivedRecord (id : Nat) (rule : Rule) (premises : List Record) : Record :=
  ⟨id, rule.conclusion, .rule rule.id (premises.map Record.id)⟩

/-- The candidate includes resolved premise payloads; whole-record equality
    against the live registry checks them. Full rule equality checks declaration.
    String syntax, capacities and host identifier validation are abstracted. -/
def stepOK (state : State) (id : Nat) (rule : Rule) (premises : List Record) : Bool :=
  decide (rule ∈ state.rules) && !rule.premises.isEmpty &&
  premises.all (fun record => decide (record ∈ state.records)) &&
  decide (premises.map Record.claim = rule.premises) &&
  state.records.all (fun record => decide (record.id ≠ id)) &&
  state.records.all (fun record =>
    decide (record.claim.key = rule.conclusion.key → record.claim.value = rule.conclusion.value))

def checkedStep (state : State) (id : Nat) (rule : Rule) (premises : List Record) : State :=
  if stepOK state id rule premises then
    { state with records := derivedRecord id rule premises :: state.records }
  else state

def releaseOK (state : State) (cert : Certificate) : Bool :=
  decide (cert.record ∈ state.records) &&
  decide (cert = certificate state cert.record) &&
  decide (cert.record.claim.key ∈ state.releaseKeys)

def release (state : State) (cert : Certificate) : Option Claim :=
  if releaseOK state cert then some cert.record.claim else none

theorem initial_invariant (session : Nat) (rules : List Rule) (keys : List Nat) :
    Invariant (initial session rules keys) := by
  intro record member
  exact False.elim (List.not_mem_nil record member)

theorem initial_consistent (session : Nat) (rules : List Rule) (keys : List Nat) :
    Consistent (initial session rules keys).records := by
  intro record member
  exact False.elim (List.not_mem_nil record member)

theorem revise_invariant (state : State) (facts : List Fact) : Invariant (revise state facts) := by
  intro record member
  rcases List.mem_map.mp member with ⟨fact, present, equal⟩
  subst record
  exact Supported.evidence fact present

theorem revision_only_evidence (state : State) (facts : List Fact)
    (record : Record) (member : record ∈ (revise state facts).records) :
    ∃ fact, fact ∈ facts ∧ record = factRecord fact := by
  rcases List.mem_map.mp member with ⟨fact, present, equal⟩
  exact ⟨fact, present, equal.symm⟩

theorem revise_consistent (state : State) (facts : List Fact)
    (consistent : EvidenceConsistent facts) : Consistent (revise state facts).records := by
  intro left leftMember right rightMember equal
  rcases List.mem_map.mp leftMember with ⟨leftFact, leftPresent, leftEqual⟩
  rcases List.mem_map.mp rightMember with ⟨rightFact, rightPresent, rightEqual⟩
  subst left; subst right
  exact consistent leftFact leftPresent rightFact rightPresent equal

theorem stepOK_sound (state : State) (id : Nat) (rule : Rule) (premises : List Record)
    (accepted : stepOK state id rule premises = true) :
    rule ∈ state.rules ∧
    (∀ record, record ∈ premises → record ∈ state.records) ∧
    premises.map Record.claim = rule.premises ∧
    (∀ record, record ∈ state.records → record.id ≠ id) ∧
    (∀ record, record ∈ state.records →
      record.claim.key = rule.conclusion.key → record.claim.value = rule.conclusion.value) := by
  simp only [stepOK, Bool.and_eq_true, decide_eq_true_eq] at accepted
  rcases accepted with ⟨⟨⟨⟨⟨declared, _⟩, members⟩, matching⟩, fresh⟩, conflict⟩
  exact ⟨declared,
    fun record present => of_decide_eq_true ((List.all_eq_true.mp members) record present),
    matching,
    fun record present => of_decide_eq_true ((List.all_eq_true.mp fresh) record present),
    fun record present => of_decide_eq_true ((List.all_eq_true.mp conflict) record present)⟩

theorem checked_step_invariant (state : State) (id : Nat) (rule : Rule) (premises : List Record)
    (invariant : Invariant state) : Invariant (checkedStep state id rule premises) := by
  by_cases accepted : stepOK state id rule premises = true
  · have valid := stepOK_sound state id rule premises accepted
    intro record member
    simp only [checkedStep, accepted, if_true, List.mem_cons] at member
    simp only [checkedStep, accepted, if_true]
    rcases member with equal | old
    · subst record
      apply Supported.derive rule valid.1
      intro claim present
      rw [← valid.2.2.1] at present
      rcases List.mem_map.mp present with ⟨premise, member, equal⟩
      rw [← equal]
      exact invariant premise (valid.2.1 premise member)
    · exact invariant record old
  · simpa [checkedStep, accepted] using invariant

theorem checked_step_consistent (state : State) (id : Nat) (rule : Rule) (premises : List Record)
    (consistent : Consistent state.records) : Consistent (checkedStep state id rule premises).records := by
  by_cases accepted : stepOK state id rule premises = true
  · have valid := stepOK_sound state id rule premises accepted
    intro left leftMember right rightMember keyEqual
    simp only [checkedStep, accepted, if_true, List.mem_cons] at leftMember rightMember
    rcases leftMember with leftNew | leftOld <;> rcases rightMember with rightNew | rightOld
    · subst left; subst right; rfl
    · subst left
      exact (valid.2.2.2.2 right rightOld keyEqual.symm).symm
    · subst right
      exact valid.2.2.2.2 left leftOld keyEqual
    · exact consistent left leftOld right rightOld keyEqual
  · simpa [checkedStep, accepted] using consistent

theorem accepted_premises_preexist (state : State) (id : Nat) (rule : Rule) (premises : List Record)
    (accepted : stepOK state id rule premises = true) :
    ∀ record, record ∈ premises → record ∈ state.records ∧ record.id ≠ id := by
  have valid := stepOK_sound state id rule premises accepted
  intro record member
  have old := valid.2.1 record member
  exact ⟨old, valid.2.2.2.1 record old⟩

theorem rejected_no_mutation (state : State) (id : Nat) (rule : Rule) (premises : List Record)
    (rejected : stepOK state id rule premises = false) :
    checkedStep state id rule premises = state := by simp [checkedStep, rejected]

theorem proposer_preserves_authority (state : State) (id : Nat) (rule : Rule) (premises : List Record) :
    let after := checkedStep state id rule premises
    after.facts = state.facts ∧ after.rules = state.rules ∧
    after.releaseKeys = state.releaseKeys ∧ after.session = state.session ∧
    after.evidenceEpoch = state.evidenceEpoch ∧ after.policyEpoch = state.policyEpoch := by
  simp only [checkedStep]
  split <;> simp

theorem policy_preserves_invariant (state : State) (keys : List Nat)
    (invariant : Invariant state) : Invariant (setPolicy state keys) := invariant

/-- Arbitrary finite serialized histories. A successful host revision carries
    its evidence-consistency precondition; proposer checks stay executable. -/
inductive Evolution (initial : State) : State → Prop where
  | initial : Evolution initial initial
  | propose (state : State) (id : Nat) (rule : Rule) (premises : List Record) :
      Evolution initial state → Evolution initial (checkedStep state id rule premises)
  | revision (state : State) (facts : List Fact) :
      Evolution initial state → EvidenceConsistent facts →
      Evolution initial (revise state facts)
  | policy (state : State) (keys : List Nat) :
      Evolution initial state → Evolution initial (setPolicy state keys)

theorem evolution_invariant (initial state : State) (initialInvariant : Invariant initial)
    (history : Evolution initial state) : Invariant state := by
  induction history with
  | initial => exact initialInvariant
  | propose state id rule premises _ ih => exact checked_step_invariant state id rule premises ih
  | revision state facts _ _ _ => exact revise_invariant state facts
  | policy state keys _ ih => exact policy_preserves_invariant state keys ih

theorem evolution_consistent (initial state : State) (initialConsistent : Consistent initial.records)
    (history : Evolution initial state) : Consistent state.records := by
  induction history with
  | initial => exact initialConsistent
  | propose state id rule premises _ ih => exact checked_step_consistent state id rule premises ih
  | revision state facts _ consistent _ => exact revise_consistent state facts consistent
  | policy _ _ _ ih => exact ih

theorem release_checks_current (state : State) (cert : Certificate)
    (accepted : releaseOK state cert = true) :
    cert.record ∈ state.records ∧ cert = certificate state cert.record ∧
    cert.record.claim.key ∈ state.releaseKeys := by
  simpa only [releaseOK, Bool.and_eq_true, decide_eq_true_eq, and_assoc] using accepted

theorem release_supported (state : State) (cert : Certificate)
    (invariant : Invariant state) (accepted : releaseOK state cert = true) :
    Supported state.facts state.rules cert.record.claim :=
  invariant cert.record (release_checks_current state cert accepted).1

theorem released_claim_registered (state : State) (cert : Certificate) (claim : Claim)
    (released : release state cert = some claim) :
    cert.record.claim = claim ∧ cert.record ∈ state.records ∧ claim.key ∈ state.releaseKeys := by
  by_cases accepted : releaseOK state cert = true
  · have equal : cert.record.claim = claim := by simpa [release, accepted] using released
    have checks := release_checks_current state cert accepted
    exact ⟨equal, checks.1, equal ▸ checks.2.2⟩
  · simp [release, accepted] at released

theorem old_evidence_certificate_rejected (state : State) (facts : List Fact) (record : Record) :
    releaseOK (revise state facts) (certificate state record) = false := by
  apply Bool.eq_false_iff.mpr
  intro accepted
  have equal := (release_checks_current _ _ accepted).2.1
  have epochs := congrArg Certificate.evidenceEpoch equal
  simp only [certificate, revise] at epochs
  exact Nat.ne_of_lt (Nat.lt_succ_self state.evidenceEpoch) epochs

theorem old_policy_certificate_rejected (state : State) (keys : List Nat) (record : Record) :
    releaseOK (setPolicy state keys) (certificate state record) = false := by
  apply Bool.eq_false_iff.mpr
  intro accepted
  have equal := (release_checks_current _ _ accepted).2.1
  have epochs := congrArg Certificate.policyEpoch equal
  simp only [certificate, setPolicy] at epochs
  exact Nat.ne_of_lt (Nat.lt_succ_self state.policyEpoch) epochs

def Holds (world : Nat → Bool) (claim : Claim) : Prop := world claim.key = claim.value

def EvidenceTrue (world : Nat → Bool) (facts : List Fact) : Prop :=
  ∀ fact, fact ∈ facts → Holds world fact.claim

def RulesSound (world : Nat → Bool) (rules : List Rule) : Prop :=
  ∀ rule, rule ∈ rules → (∀ claim, claim ∈ rule.premises → Holds world claim) →
    Holds world rule.conclusion

theorem support_has_evidence_root (facts : List Fact) (rules : List Rule)
    (nonemptyRules : ∀ rule, rule ∈ rules → rule.premises ≠ []) (claim : Claim)
    (supported : Supported facts rules claim) : ∃ fact, fact ∈ facts := by
  induction supported with
  | evidence fact member => exact ⟨fact, member⟩
  | derive rule member _ ih =>
    cases premisesEqual : rule.premises with
    | nil => exact False.elim (nonemptyRules rule member premisesEqual)
    | cons head tail => exact ih head (by simp [premisesEqual])

/-- Conditional truth: evidence and declared rules must be true in THIS world.
    The checker does not establish existence of such a world or authenticate it. -/
theorem supported_semantically_sound (facts : List Fact) (rules : List Rule)
    (world : Nat → Bool) (evidenceTrue : EvidenceTrue world facts)
    (rulesSound : RulesSound world rules) (claim : Claim)
    (supported : Supported facts rules claim) : Holds world claim := by
  induction supported with
  | evidence fact member => exact evidenceTrue fact member
  | derive rule member _ ih => exact rulesSound rule member ih

theorem release_semantically_sound (state : State) (cert : Certificate) (world : Nat → Bool)
    (invariant : Invariant state) (accepted : releaseOK state cert = true)
    (evidenceTrue : EvidenceTrue world state.facts) (rulesSound : RulesSound world state.rules) :
    Holds world cert.record.claim :=
  supported_semantically_sound state.facts state.rules world evidenceTrue rulesSound
    cert.record.claim (release_supported state cert invariant accepted)

/-- A constructive counterexample to the unconditional external-truth claim. -/
theorem support_does_not_imply_external_truth :
    ∃ facts rules claim world, Supported facts rules claim ∧ ¬ Holds world claim := by
  let fact : Fact := ⟨0, ⟨0, true⟩⟩
  refine ⟨[fact], [], fact.claim, (fun _ => false), ?_, ?_⟩
  · exact Supported.evidence fact (by simp)
  · change ¬ (false = true)
    exact Bool.false_ne_true

#print axioms initial_invariant
#print axioms initial_consistent
#print axioms revise_invariant
#print axioms revision_only_evidence
#print axioms revise_consistent
#print axioms stepOK_sound
#print axioms checked_step_invariant
#print axioms checked_step_consistent
#print axioms accepted_premises_preexist
#print axioms rejected_no_mutation
#print axioms proposer_preserves_authority
#print axioms policy_preserves_invariant
#print axioms evolution_invariant
#print axioms evolution_consistent
#print axioms release_checks_current
#print axioms release_supported
#print axioms released_claim_registered
#print axioms old_evidence_certificate_rejected
#print axioms old_policy_certificate_rejected
#print axioms support_has_evidence_root
#print axioms supported_semantically_sound
#print axioms release_semantically_sound
#print axioms support_does_not_imply_external_truth

end SupportedState
