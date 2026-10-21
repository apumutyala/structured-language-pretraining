import SupportedState

/- Bounded CPU test adapter only. No network, auth, Python deserialization,
   learned model, or production certificate API. -/
open SupportedState
namespace SupportedStateCli

def keys : List Nat := [0, 1, 2]

def empty (rules : List Rule) (policy : List Nat := keys) : State :=
  initial 7 rules policy

def claimOf (index : Nat) : Claim := ⟨index / 2, index % 2 == 1⟩

def claimsMask (records : List Record) : Nat :=
  records.foldl (fun mask record =>
    mask ||| 2 ^ (record.claim.key * 2 + if record.claim.value then 1 else 0)) 0

def releaseMask (state : State) : Nat :=
  claimsMask (state.records.filter (fun record => releaseOK state (certificate state record)))

def output (state : State) (accepted stale : Bool) : String :=
  String.intercalate " " [if accepted then "1" else "0", toString state.evidenceEpoch,
    toString state.policyEpoch, toString state.records.length, toString (claimsMask state.records),
    toString (releaseMask state), if stale then "1" else "0"]

def add (state : State) (fact : Fact) : State := revise state (state.facts ++ [fact])

def recordAt (state : State) (id : Nat) : List Record :=
  state.records.filter (fun record => record.id == id)

def single (premise conclusion evidence : Nat) : String := Id.run do
  let rule : Rule := ⟨0, [claimOf premise], claimOf conclusion⟩
  let mut state := empty [rule]
  for key in keys do
    let value := evidence / 3 ^ key % 3
    if value > 0 then state := add state ⟨key, ⟨key, value == 2⟩⟩
  let premises := recordAt state (premise / 2)
  let accepted := stepOK state 100 rule premises
  return output (checkedStep state 100 rule premises) accepted false

def revisions (operation policy : Nat) : String := Id.run do
  let a : Claim := ⟨0, true⟩
  let b : Claim := ⟨1, true⟩
  let c : Claim := ⟨2, true⟩
  let ab : Rule := ⟨0, [a], b⟩
  let bc : Rule := ⟨1, [b], c⟩
  let mut state := add (empty [ab, bc]) ⟨0, a⟩
  state := checkedStep state 100 ab (recordAt state 0)
  state := checkedStep state 101 bc (recordAt state 100)
  let cert := certificate state (derivedRecord 101 bc (recordAt state 100))
  state := match operation with
    | 0 => revise state [⟨0, a⟩]
    | 1 => revise state [⟨0, ⟨0, false⟩⟩]
    | 2 => revise state [⟨0, c⟩]
    | 3 => revise state []
    | 4 => add state ⟨1, ⟨1, true⟩⟩
    | 5 => setPolicy state (keys.filter (fun key => policy / 2 ^ key % 2 == 1))
    | _ => setPolicy state keys
  return output state true (releaseOK state cert)

def conjunction (mode : Nat) : String := Id.run do
  let a : Claim := ⟨0, true⟩
  let b : Claim := ⟨1, false⟩
  let c : Claim := ⟨2, true⟩
  let rule : Rule := ⟨0, [a, b], c⟩
  let mut state := add (add (empty [rule]) ⟨0, a⟩) ⟨1, b⟩
  if mode == 8 then state := add state ⟨2, ⟨2, false⟩⟩
  let first := recordAt state 0
  let second := recordAt state 1
  let premises := match mode with
    | 1 => second ++ first
    | 2 => first ++ first
    | 3 => first
    | 4 => [⟨0, a, .evidence 99⟩] ++ second
    | 9 => [⟨9, a, .evidence 9⟩] ++ second
    | _ => first ++ second
  let proposedRule := if mode == 6 then { rule with conclusion := a }
    else if mode == 7 then { rule with id := 99 } else rule
  let id := if mode == 5 then 0 else 100
  let accepted := stepOK state id proposedRule premises
  return output (checkedStep state id proposedRule premises) accepted false

def parseNat (token : String) : Except String Nat := do
  if token.isEmpty || !(token.data.all (fun c => decide ('0' ≤ c ∧ c ≤ '9'))) then
    throw "unsigned decimal required"
  match token.toNat? with
  | some n => return n
  | none => throw "invalid integer"

def handle (line : String) : Except String String := do
  if line.length > 128 then throw "fixture line too long"
  let words := (line.split Char.isWhitespace).filter (fun word => !word.isEmpty)
  match words with
  | ["S", premise, conclusion, evidence] =>
    let p ← parseNat premise
    let c ← parseNat conclusion
    let e ← parseNat evidence
    if p < 6 && c < 6 && e < 27 then return single p c e
    else throw "single fixture range"
  | ["R", operation, policy] =>
    let op ← parseNat operation
    let policy ← parseNat policy
    if op < 7 && policy < 8 then return revisions op policy
    else throw "revision fixture range"
  | ["J", mode] =>
    let mode ← parseNat mode
    if mode < 10 then return conjunction mode
    else throw "conjunction fixture range"
  | _ => throw "expected S/R/J fixture"

partial def loop (input output : IO.FS.Stream) : IO Unit := do
  let line ← input.getLine
  if line.isEmpty then return
  output.putStrLn (match handle line with | .ok answer => answer | .error reason => "ERROR " ++ reason)
  output.flush
  loop input output

end SupportedStateCli

def main : IO Unit := do
  SupportedStateCli.loop (← IO.getStdin) (← IO.getStdout)
