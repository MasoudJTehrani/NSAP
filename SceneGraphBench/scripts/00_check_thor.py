#!/usr/bin/env python
"""Step 2 smoke test: confirm AI2-THOR renders frames + instance masks headlessly."""

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering


def main():
    c = Controller(platform=CloudRendering, scene="FloorPlan1",
                   width=300, height=300, renderInstanceSegmentation=True)
    e = c.step("Pass")
    print("frame:          ", e.frame.shape)
    print("objects:        ", len(e.metadata["objects"]))
    print("instance masks: ", len(e.instance_masks))
    c.stop()
    print("OK — AI2-THOR renders headlessly and exposes ground-truth masks.")


if __name__ == "__main__":
    main()
