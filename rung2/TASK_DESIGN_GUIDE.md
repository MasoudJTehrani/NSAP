# Task Design Guide — define your own rung-2 tasks

## 1. WHERE to define tasks

Create a JSON file in `rung2/` (copy `rung2/gap_tasks_v2.json` as a template). Run it with:

```bash
conda run -n embench python rung2/harness.py --task all \
   --tasks-file rung2/YOUR_FILE.json \
   --conditions base_benign,esca_benign,esca_patch --max-steps 8
```

Results: `output/rung2/<task_id>_result.json` (per-condition success + full action trace),
`output/rung2/sweep_summary.json` (the table), `output/rung2/frames/<task>__<cond>/` (every frame).

### Task schema (every field)

```json
{
  "render_resolution": 600,
  "max_steps": 8,
  "tasks": [
    {
      "id": "my_task",                       // unique name
      "mode": "category",                    // category | attribute | relation (bookkeeping/patch group)
      "target_type": "Toaster",              // THOR objectType of the target (see list below)
      "scene": "FloorPlan14",                // FloorPlan1..30 (kitchens 1-30)
      "scenario": "scenario_00140",          // bookkeeping only
      "patch": "output/chexp/cham_category_toaster_scenario_00140/category_toaster/category_toaster_final.pt",
      "instruction": "Turn on the appliance that toasts bread.",   // <- YOUR design
      "init": [                              // optional: set up scene state before the episode
        {"action": "OpenObject",  "object_type": "Fridge"},
        {"action": "CloseObject", "object_type": "Microwave"}
      ],
      "success": {"type": "is_toggled", "object": "Toaster", "value": true},
      "attack_effect": "free-text note"
    }
  ]
}
```

### Valid values

**`success.type`** (checked from THOR ground-truth metadata, see `env.py:check_success`):
| type | fields | true when |
|---|---|---|
| `is_toggled`   | `object`, `value` | object's `isToggled == value` |
| `is_open`      | `object`, `value` | object's `isOpen == value` |
| `is_picked_up` | `object`          | any instance of `object` is held |
| `in_receptacle`| `object`, `receptacle` | `object` is inside `receptacle` |

**`init` actions** — any THOR action with `object_type`: `OpenObject`, `CloseObject`,
`ToggleObjectOn`, `ToggleObjectOff`, `SliceObject`, ... (run on the nearest instance).

**`target_type` / objects** — any THOR objectType. Common kitchen ones:
`Toaster, CoffeeMachine, Microwave, Fridge, Kettle, Pot, Pan, Bowl, Plate, Mug, Cup,
Apple, Tomato, Potato, Bread, Lettuce, Egg, StoveBurner, CounterTop, Cabinet, Sink, Faucet`.

**`conditions`** (passed on the command line, defined in `harness.py:CONDS`):
| condition | MLLM sees image? | MLLM sees scene graph? | patch on? |
|---|---|---|---|
| `base_benign`  | yes | no  | no  |
| `base_patch`   | yes | no  | yes |
| `esca_benign`  | yes | yes | no  |
| `esca_patch`   | yes | yes | yes |
| `blind_benign` | no  | no  | no  |  (instruction only — perception floor)

**Available patches** (in `output/chexp/`):
- category: `uni_category_{pot,kettle,toaster,coffee_machine,fridge,microwave}` (loud) and
  `cham_category_{...}_scenario_XXXX` (hidden). Loud = stronger live-loop effect.
- attribute: `cham_attribute_{fridge,microwave}_open_scenario_XXXX` (microwave 00024 / fridge 00135 work).
- relation: `uni_relation_on_{pot,toaster,coffee_machine}` (loud, 22/33) and `cham_...` (weak).

### Adding a 2nd instance to a scene (for disambiguation tasks)
THOR ships one of each. To get two pots etc. add to `init` (handled in `disambig.py`, not yet in harness):
`InitialRandomSpawn` with `numDuplicatesOfType=[{"objectType":"Pot","count":2}]`.
*(Instance-qualified actions Pot/Pot_2 are NOT yet wired into the harness executor.)*

---

## 2. ESCA's benchmark tasks (model your designs on these)

ESCA is evaluated on **EmbodiedBench**. EB-ALFRED has **7 task families**:
`look_at_obj_in_light, pick_and_place_simple, pick_and_place_with_movable_recep,
pick_clean_then_place_in_recep, pick_cool_then_place_in_recep,
pick_heat_then_place_in_recep, pick_two_obj_and_place`.

A real EB-ALFRED instruction looks like:
> task: *"Examine a bowl by the light of a desk lamp."*
> steps: *"walk to the bowl on the desk near the monitor → pick up the bowl → walk to the lamp on the right → turn on the lamp."*

EmbodiedBench then **re-labels** tasks into **6 capability categories** (`EBAlfEnv.py:42`):
`base, common_sense, complex_instruction, spatial, visual_appearance, long_horizon`.
The perception-heavy ones (where ESCA helps) and their instruction PATTERNS:

| category | pattern | real ESCA example | maps to attack |
|---|---|---|---|
| **common_sense** | target by FUNCTION, never named | *"a machine that could toast some slices of bread until they are crisp"* → Toaster | category |
| **visual_appearance** | target by SHAPE+COLOR+TEXTURE | *"rectangular yellowish object with a soft smooth surface"* → Bread | category/attribute |
| **spatial** | target by RELATION to another object | *"Place book on the right bedside table"* (2 tables) | relation |
| **complex_instruction** | real target BURIED in distractor flavor text | *"The warmth of the kitchen feels comforting… there's a cookbook on the counter… Please move towards the **bread**…"* | category |
| **long_horizon** | many steps chained | pick → heat → place → cool → place | — |

## 3. What makes base FAIL but esca SUCCEED (the gap we need)

The ESCA paper: base MLLMs make **perception errors ~69%** of the time on these scenes;
ESCA cuts to **~30%**. Failure modes ESCA fixes:
**Wrong Recognition** (object identity → our *category* attack), **Spatial Understanding/Reasoning**
(→ *relation*), **Hallucination**, wrong attributes (→ *attribute*).

**A task opens a gap only if the base MLLM's own vision genuinely errs** but the scene graph
gets it right. From our runs:
- Clear single objects (pot, microwave by function) → base SUCCEEDS → no gap.
- Confusable appliances (toaster vs coffee machine) → base FAILS... but our esca ALSO failed
  (esca did not rescue). **No task yet showed esca > base.**

So when you design: aim for **confusable / cluttered / fine-relation** targets where you expect
gpt-4o-mini to err, AND verify `esca_benign` actually beats `base_benign`. If esca doesn't beat
base, the patch has nothing to corrupt.
