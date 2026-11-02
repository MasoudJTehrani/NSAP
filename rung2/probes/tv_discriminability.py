"""Can the planner still tell the TV is on, from the candidate far poses?

The whole plan rests on one premise: at distance the MLLM cannot judge the screen
state, so it must fall back on the symbolic attribute. That premise is NOT obvious,
because a switched-on screen is BRIGHT, and brightness survives downsampling far
better than detail does.

Test it directly and cheaply before spending hours on a benchmark. For each candidate
pose we render the frame twice, TV on and TV off, and ask the same model the planner
uses whether the television is on. If it answers correctly at that distance, distance
does not hide the attribute and the task design will not work.

Cost: 2 API calls per pose per scene.
"""
import base64, json, math, os, sys
sys.path.insert(0, "rung2")
import ai2thor.controller
from PIL import Image
import io

POSES = "output/probes/tv_poses.json"
OUTDIR = "output/probes/tvshots"
RES = 500
MODEL = os.environ.get("NSAP_PROBE_MODEL", "gpt-4o")

def ask(client, img, prompt):
    buf = io.BytesIO(); Image.fromarray(img).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    r = client.chat.completions.create(model=MODEL, max_tokens=10, temperature=0,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}])
    return r.choices[0].message.content.strip().lower()

def main():
    from openai import OpenAI
    client = OpenAI()
    os.makedirs(OUTDIR, exist_ok=True)
    data = json.load(open(POSES))
    c = ai2thor.controller.Controller(scene="FloorPlan201", width=RES, height=RES,
                                      renderInstanceSegmentation=True)
    Q = ("Look at the television in this image. Is its screen switched ON (showing a "
         "lit picture) or OFF (dark)? Answer with exactly one word: on or off.")
    rows = []
    for scene, info in data.items():
        cands = info.get("smallest") or []
        if not cands:
            continue
        pick = cands[0]                                  # the smallest, hardest case
        c.reset(scene=scene)
        tv = [o for o in c.last_event.metadata["objects"] if o["objectType"] == "Television"][0]
        for state in ("on", "off"):
            c.step(action="ToggleObjectOn" if state == "on" else "ToggleObjectOff",
                   objectId=tv["objectId"], forceAction=True)
            c.step(action="Teleport", position=pick["position"],
                   rotation={"x": 0, "y": pick["rotation"], "z": 0},
                   horizon=pick["horizon"], standing=True)
            frame = c.last_event.frame
            Image.fromarray(frame).save(f"{OUTDIR}/{scene}_{state}.png")
            try:
                a = ask(client, frame, Q)
            except Exception as e:
                a = f"ERROR {repr(e)[:60]}"
            correct = a.startswith(state)
            rows.append((scene, pick["dist"], pick["area"], state, a, correct))
            print(f"  {scene:14s} d={pick['dist']}m area={pick['area']:>6}px  "
                  f"truth={state:3s} answer={a:12s} {'CORRECT' if correct else 'wrong'}", flush=True)
    n = len(rows); ok = sum(1 for r in rows if r[5])
    print(f"\nmodel judged the screen state correctly in {ok}/{n} images "
          f"({ok/max(n,1):.0%})")
    print("VERDICT: distance does NOT hide the attribute; pick another lever."
          if ok/max(n,1) > 0.7 else
          "VERDICT: the state is genuinely hard to read at these poses; the design can work.")
    json.dump([{"scene":r[0],"dist":r[1],"area":r[2],"truth":r[3],"answer":r[4],"correct":r[5]}
               for r in rows], open(f"{OUTDIR}/result.json","w"), indent=2)
    c.stop()

if __name__ == "__main__":
    main()
