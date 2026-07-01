# Gap-tasks: referential instructions where the **base MLLM fails but ESCA rescues**

**Goal.** Find instructions that gpt-4o-mini cannot solve from its own vision, but *can* solve
once ESCA's scene graph grounds the perception — so that corrupting that scene graph with our
(unchanged) patch makes the **full multimodal pipeline** fail. Validation chain per task:

```
base_benign (image only)      -> FAIL     (MLLM mis-identifies the referent)
esca_benign (image + SG)      -> SUCCESS  (scene graph grounds the right object)  => the GAP
esca_patch  (SG corrupted)    -> FAIL     (grounding broken -> back to mis-identify)
```

## Design principle (from the ESCA paper + EmbodiedBench)

ESCA fixes **perception errors** — base MLLMs err ~69% of the time on these scenes, ESCA cuts it
to ~30%. The failure modes it fixes map onto our three attacks:

| ESCA failure mode fixed | our attack | EB capability category |
|---|---|---|
| **Wrong Recognition** (object identity) | **category** | common_sense / visual_appearance |
| **Spatial Understanding/Reasoning** | **relation** | spatial |
| wrong attribute/state | **attribute** | visual_appearance |

**The lever:** the instruction **never names the target**. It describes it by FUNCTION,
APPEARANCE, or RELATION, among **confusable distractors**, so the MLLM must recognize *which*
object it is. That recognition is exactly what the scene graph supplies and the patch corrupts.

---

## A. Category — by FUNCTION (common_sense pattern)
*"a machine that could toast some slices of bread until they are crisp" → Toaster*

1. **toast_fn** — FloorPlan14 (toaster **+ coffee machine** distractor)
   > "Turn on the small countertop appliance that browns slices of bread until they are crisp. Leave every other appliance switched off."
   - target Toaster · action turn-on · success `is_toggled Toaster=true` · patch `category_toaster`
   - *why base fails:* toaster vs coffee-machine vs kettle are confusable small appliances; base already turned on the **microwave** here. *esca:* grounds the toaster. *patch:* toaster→other → can't pick it.

2. **coffee_fn** — FloorPlan13 (coffee machine + toaster/kettle)
   > "Switch on the machine people use to brew a fresh cup of coffee in the morning."
   - target CoffeeMachine · turn-on · `is_toggled CoffeeMachine=true` · patch `category_coffee_machine`

3. **boil_vessel_fn** — FloorPlan1 (pot **+ pan + bowl** distractors)
   > "Pick up the deep round metal vessel used to boil pasta or simmer a soup — not a shallow pan or a small bowl."
   - target Pot · pick-up · `is_picked_up Pot` · patch `category_pot`
   - *why base fails:* pot vs pan vs bowl are visually similar round vessels.

4. **reheat_fn** — FloorPlan4 (microwave + other appliances)
   > "Open the boxy appliance that reheats leftovers using microwaves so I can put food inside."
   - target Microwave · open · `is_open Microwave=true` · patch `category_microwave`

## B. Category — by APPEARANCE (visual_appearance pattern)
*"rectangular yellowish object with a soft smooth surface" → Bread*

5. **toast_appear** — FloorPlan14
   > "Turn on the rectangular metal device with two slots on top and a push-down lever on its side."
   - target Toaster · turn-on · `is_toggled Toaster=true` · patch `category_toaster`

6. **kettle_appear** — FloorPlan (kettle + pot/coffee machine)
   > "Pick up the shiny metallic container with a spout and a handle, used for boiling water."
   - target Kettle · pick-up · `is_picked_up Kettle` · patch `category_kettle`

## C. Relation — by SPATIAL relation (spatial pattern) — needs 2-instance infra
*"Place book on right bedside table" (2 tables)*

7. **on_stove_rel** — FloorPlan1, **two pots spawned** (one on stove, one on counter)
   > "Pick up the cooking pot that is sitting on the stove burner — not the one resting on the counter."
   - target on-stove Pot (instance-specific) · pick-up · patch `relation_on_pot`
   - *needs* instance-qualified actions (`Pot`/`Pot_2`) in the executor.

## D. Negation / exclusion (complex_instruction pattern) — hardest
8. **except_toaster** — FloorPlan14
   > "Turn on every small appliance on the counter **except** the one that toasts bread."
   - success: Toaster `isToggled=false` AND the others `isToggled=true` · patch `category_toaster`
   - *requires* identifying the toaster (to exclude it); patch makes it indistinguishable → it gets turned on too.

---

## Status / how to run
Type-level tasks (A,B,D) run on the existing `harness.py` (no instance infra).
Relation (C) needs the instance-qualified executor (not yet built).

```
conda run -n embench python rung2/harness.py --task all \
   --tasks-file rung2/gap_tasks_v2.json \
   --conditions base_benign,esca_benign,esca_patch --max-steps 8
```
Keep only tasks where `base_benign=FAIL` (a real gap); discard any where base already succeeds.
