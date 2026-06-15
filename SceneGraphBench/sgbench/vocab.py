"""
Vocabularies and AI2-THOR objectType -> natural-language word mapping.

Everything here is data, not logic — edit these tables to retarget the benchmark
to different categories, attributes, or relations.
"""

import re

# ── Category targets ──────────────────────────────────────────────────────────
# Objects an embodied task can hinge on. The default pilot set is the cluster a
# CLIP-based scene-graph model reliably recognizes on AI2-THOR renders AND that
# kitchen tasks depend on (heat/cook/boil/toast/coffee). Extend freely.
PILOT_CATEGORIES = {'Pot', 'Kettle', 'Toaster', 'CoffeeMachine'}

TARGET_CATEGORIES = {
    'Pot', 'Kettle', 'Toaster', 'CoffeeMachine', 'Microwave', 'Fridge',
    'Mug', 'Cup', 'Bowl', 'Plate', 'Pan',
    'Apple', 'Bread', 'Tomato', 'Lettuce', 'Potato', 'Egg',
    'Knife', 'ButterKnife', 'Fork', 'Spoon', 'Spatula',
    'WineBottle', 'SoapBottle', 'Vase', 'HousePlant', 'Book', 'Laptop', 'CellPhone',
}

# ── Attributes (object STATES) ────────────────────────────────────────────────
# metadata bool flag -> (capability flag that makes it meaningful, word_if_true,
# word_if_false_or_None). Ground truth comes straight from the simulator state.
ATTRIBUTE_FLAGS = {
    'isOpen':             ('openable',         'open',   'closed'),
    'isToggled':          ('toggleable',       'on',     'off'),
    'isSliced':           ('sliceable',        'sliced', None),
    'isCooked':           ('cookable',         'cooked', None),
    'isDirty':            ('dirtyable',        'dirty',  'clean'),
    'isFilledWithLiquid': ('canFillWithLiquid','filled', None),
    'isBroken':           ('breakable',        'broken', None),
}

# ── Relations ─────────────────────────────────────────────────────────────────
# A parentReceptacle of one of these enclosing types is reported as "in",
# otherwise "on". Distance-based "near" is added by scene_graph.py.
ENCLOSING_RECEPTACLES = {
    'Fridge', 'Microwave', 'Cabinet', 'Drawer', 'Box', 'Safe',
    'Sink', 'SinkBasin', 'Pot', 'Pan', 'Bowl', 'Cup', 'Mug',
}

# ── objectType -> word ────────────────────────────────────────────────────────
_SPECIAL = {
    'CounterTop': 'countertop', 'SoapBottle': 'soap bottle', 'SprayBottle': 'spray bottle',
    'GarbageCan': 'garbage can', 'DiningTable': 'dining table', 'SideTable': 'side table',
    'CoffeeTable': 'coffee table', 'StoveBurner': 'stove burner', 'StoveKnob': 'stove knob',
    'ButterKnife': 'butter knife', 'PepperShaker': 'pepper shaker', 'SaltShaker': 'salt shaker',
    'DishSponge': 'dish sponge', 'WineBottle': 'wine bottle', 'CellPhone': 'cellphone',
    'RemoteControl': 'remote control', 'HousePlant': 'house plant', 'CreditCard': 'credit card',
    'TissueBox': 'tissue box', 'LightSwitch': 'light switch', 'SinkBasin': 'sink',
    'Glassbottle': 'glass bottle', 'CoffeeMachine': 'coffee machine',
}


def thor_type_to_word(otype: str) -> str:
    """'AlarmClock' -> 'alarm clock'.  Special-cased where camelCase splitting is wrong."""
    if otype in _SPECIAL:
        return _SPECIAL[otype]
    return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', otype).lower()
