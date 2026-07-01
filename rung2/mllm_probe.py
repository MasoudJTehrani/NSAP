#!/usr/bin/env python
"""
Does our (SGClip-tuned) patch also fool the MLLM's OWN vision?

This is the hinge for "the whole multimodal pipeline fails": we attack SGClip, but
gpt-4o-mini sees the same image directly. Render the on-stove pot, inject a patch,
and ask gpt-4o-mini itself what is on the stove -- clean vs patched. If its answer
flips (no longer "a pot"), the patch transfers to the MLLM and the full pipeline can
fail. If it still says "pot", the MLLM's vision is redundant and absorbs the attack.

Tests both the LOUD pot patch (most disruptive) and the hidden chameleon.

conda run -n embench python rung2/mllm_probe.py
"""
import os, sys, base64, io
sys.path.insert(0, "/home/usi/NSAP/rung2")
import numpy as np
from PIL import Image
from harness import load_patch, NSAP   # also sets OPENAI_API_KEY from ESCA/.openai_key
from env import Rung2Env
from flip_rate import inject
from openai import OpenAI

PATCHES = {
    "loud":      "output/chexp/uni_category_pot/category_pot/category_pot_final.pt",
    "chameleon": "output/chexp/cham_category_pot_scenario_00087/category_pot/category_pot_final.pt",
}
QUESTION = ("Look only at the stove/cooktop area. What object is sitting on the stove "
            "burner? Answer in 1-2 words (e.g. 'a pot', 'a pan', 'nothing', 'unclear').")


def on_stove(o):
    return any('StoveBurner' in (p or '') for p in (o.get('parentReceptacles') or []))


def ask(client, img_np):
    b = io.BytesIO(); Image.fromarray(img_np).save(b, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
    r = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0, max_tokens=20,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": QUESTION},
            {"type": "image_url", "image_url": {"url": url}}]}])
    return r.choices[0].message.content.strip()


def main():
    client = OpenAI()
    env = Rung2Env("FloorPlan1", resolution=600)
    env.c.step(action="InitialRandomSpawn", randomSeed=0, forceVisible=True,
               numDuplicatesOfType=[{"objectType": "Pot", "count": 2}])
    target = next(o for o in env.objects() if o['objectType'] == 'Pot' and on_stove(o))
    frame, mask, bbox = env.view_target(target['objectId'])
    if frame is None:
        print("on-stove pot not visible"); env.stop(); return
    Image.fromarray(frame).save("output/rung2/mllm_probe_clean.png")
    print(f"gpt-4o-mini, 'what is on the stove?'\n  CLEAN          : {ask(client, frame)}")
    for name, pth in PATCHES.items():
        if not os.path.exists(os.path.join(NSAP, pth)):
            print(f"  {name:14}: (patch missing)"); continue
        patched = inject(frame, mask, bbox, load_patch(os.path.join(NSAP, pth)))
        Image.fromarray(patched).save(f"output/rung2/mllm_probe_{name}.png")
        print(f"  PATCH ({name:9}): {ask(client, patched)}")
    env.stop()


if __name__ == "__main__":
    main()
