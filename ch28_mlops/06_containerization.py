# %% [markdown]
# # Section 28.6 Example 28-10: the environment an image pins and its build gate
#
# The code cell is Example 28-10 of Section 28.6. It prints the Python version and the platform it
# runs on, then the versions of the five packages that requirements.txt in this folder pins for the
# serving image, and then runs the build gate, which loads 04_serving.py the way a server imports it
# and checks its label for one held-out tumor of the breast cancer Wisconsin data. The Dockerfile in
# this folder runs this script as a build step; the notebook builds no image.

# %%
import sys, platform, importlib, runpy

print("environment a container pins (so it is identical everywhere):")
print(f"  python   : {sys.version.split()[0]}")
print(f"  platform : {platform.system()} {platform.machine()}")
pkgs = ["numpy", "scikit-learn", "fastapi", "uvicorn", "pydantic"]
pins = []
for name in pkgs:
    mod = importlib.import_module("sklearn" if name == "scikit-learn" else name)
    ver = getattr(mod, "__version__", "?")
    pins.append(f"{name}=={ver}")
    print(f"  {name:13s}: {ver}")
print("  pinned requirements (baked into the image):")
print("    " + "  ".join(pins))

def smoke_test():                                           # build gate: real serving code, one known tumor
    ns = runpy.run_path("04_serving.py", run_name="serving")  # as a server imports it; skips the demo
    out = ns["predict"](ns["PredictRequest"](features=ns["Xte"][3].tolist()))   # held-out tumor Xte[3]
    assert out["label"] == int(ns["yte"][3]), out          # a wrong label fails the build
    return out, int(ns["yte"][3])

print("\ncontainer smoke test (run at build time; an import error or a wrong label stops the build):")
out, truth = smoke_test()
print(f"  load 04_serving.py -> predict(held-out Xte[3]) -> {out}")
print(f"  true label {truth} -> test passed")
print("\na container image = your code + its pinned environment, built once and run the same on a")
print("laptop, a CI runner, and a production server.")

print("\nAll containerization checks ran.")

