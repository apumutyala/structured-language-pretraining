import WorkflowKernel

open WorkflowKernel

namespace WorkflowKernelCli

def parseNat (token : String) : Except String Nat := do
  if token.isEmpty || !(token.data.all (fun ch => decide ('0' ≤ ch ∧ ch ≤ '9'))) then
    throw "expected unsigned decimal integer"
  match token.toNat? with
  | some n => return n
  | none => throw "invalid unsigned decimal integer"

def boundedFin (value bound : Nat) (name : String) : Except String (Fin bound) :=
  if valid : value < bound then .ok ⟨value, valid⟩
  else .error (name ++ " out of range")

def parsePrior (value : Nat) : Except String PriorMask := do
  let mask ← boundedFin value 256 "prior mask"
  if nonzero : mask.val ≠ 0 then return ⟨mask, nonzero⟩
  else throw "prior mask must be nonempty"

def parseAction (tag : Nat) : Except String Action :=
  match tag with
  | 0 => .ok .wait
  | 1 => .ok (.submit ⟨0, by decide⟩)
  | 2 => .ok (.submit ⟨1, by decide⟩)
  | 3 => .ok (.submit ⟨2, by decide⟩)
  | _ => .error "action out of range"

def parseObservation (tag x y : Nat) : Except String Observation := do
  match tag with
  | 0 =>
    if x = 0 ∧ y = 0 then return .timeout
    else throw "TIMEOUT requires x=y=0"
  | 1 =>
    if x = 0 ∧ y = 0 then return .untrusted
    else throw "UNTRUSTED requires x=y=0"
  | 2 =>
    let index ← boundedFin x 3 "receipt index"
    if y < 2 then return .receipt index (y == 1)
    else throw "receipt success out of range"
  | 3 =>
    let mask ← boundedFin x 8 "parity mask"
    if y < 2 then return .parity mask (y == 1)
    else throw "parity rhs out of range"
  | 4 =>
    let mask ← boundedFin x 8 "count mask"
    let count ← boundedFin y (popcount mask.val + 1) "count value"
    return .count mask count
  | _ => throw "observation tag out of range"

def boolDigit (value : Bool) : String := if value then "1" else "0"

def renderResult (result : Result) : String :=
  String.intercalate " " [toString (supportMask result.state.memory),
    toString result.state.grants.val, toString result.state.budget,
    boolDigit result.accepted, boolDigit result.contradiction,
    boolDigit result.actionPermitted, toString (legalSubmitMask result.state)]

def handle (line : String) : Except String String := do
  let words := (line.split Char.isWhitespace).filter (fun word => !word.isEmpty)
  match words with
  | ["ASSERT", memory, tag, x, y] =>
    let memory ← parseNat memory
    let tag ← parseNat tag
    let x ← parseNat x
    let y ← parseNat y
    let mask ← boundedFin memory 256 "assertion memory mask"
    let observation ← parseObservation tag x y
    return "ASSERT " ++ boolDigit (groundedObservation mask observation)
  | [prior, grants, budget, action, tag, x, y, candidate] =>
    let prior ← parseNat prior >>= parsePrior
    let grants ← parseNat grants >>= fun n => boundedFin n 8 "grants"
    let budget ← parseNat budget
    if budget > 9223372036854775807 then throw "budget out of range"
    let action ← parseNat action >>= parseAction
    let tag ← parseNat tag
    let x ← parseNat x
    let y ← parseNat y
    let observation ← parseObservation tag x y
    let candidate ← parseNat candidate >>= fun n => boundedFin n 256 "candidate mask"
    return renderResult (run ⟨prior, grants, budget, action, observation, candidate⟩)
  | _ => throw "expected eight integers, or ASSERT followed by four integers"

partial def loop (input output : IO.FS.Stream) : IO Unit := do
  let line ← input.getLine
  if line.isEmpty then return
  let answer := match handle line with
    | .ok answer => answer
    | .error reason => "ERROR " ++ reason
  output.putStrLn answer
  output.flush
  loop input output

end WorkflowKernelCli

def main : IO Unit := do
  WorkflowKernelCli.loop (← IO.getStdin) (← IO.getStdout)
