#!/bin/bash
# Section 3.8.5, Data versioning: the chapter's DVC block, run end to end on the Titanic table.
# It follows the DVC Get Started guide (doc.dvc.org/start) and adds one commit, of .dvc/config.
# Needs Git, DVC (installed in its own environment, as the guide advises: uv tool install dvc,
# or pipx install dvc), and a Python with seaborn and pandas (the book's environment) to write
# and change the CSV. Set PY to that Python if it is not the first python on your PATH.
# usage: bash dvc_versioning_demo.sh
# It works in a fresh temporary folder, so it never touches this one, and it uses /tmp/dvcstore
# as the remote, as the guide does; it stops if that folder already exists, and removes it at the end.
set -euo pipefail
PY="${PY:-python}"
if [ -e /tmp/dvcstore ]; then echo "/tmp/dvcstore already exists; remove it first"; exit 1; fi
WORK="$(mktemp -d)"; cd "$WORK"; mkdir data
echo "working in a temporary folder | DVC $(dvc --version) | $(git --version)"

# The chapter's Titanic table, 891 passengers by 15 columns, saved as data/titanic.csv.
"$PY" -c 'import seaborn as sns; sns.load_dataset("titanic").to_csv("data/titanic.csv", index=False)'

git init -q
dvc init                                        # creates DVC's settings in .dvc/, staged for Git
git commit -q -m "Initialize DVC"
dvc add data/titanic.csv                        # hash the file into DVC's cache
git add data/titanic.csv.dvc data/.gitignore    # the pointer file and the ignore entry
git commit -q -m "Track the Titanic data"
echo "--- the pointer Git stores:"; cat data/titanic.csv.dvc
echo "--- the files Git tracks:"; git ls-files
mkdir /tmp/dvcstore                             # a local folder to serve as the remote
dvc remote add -d myremote /tmp/dvcstore        # point DVC at that folder
git commit -q .dvc/config -m "Configure the remote" # the remote's address, kept in Git
dvc push                                        # copy the cached data to the remote

# later, after the CSV changes: here the target-in-disguise column alive is dropped (3.10).
"$PY" -c 'import pandas as pd; pd.read_csv("data/titanic.csv").drop(columns=["alive"]).to_csv("data/titanic.csv", index=False)'
dvc add data/titanic.csv                        # hash the new version
dvc push
git commit -q data/titanic.csv.dvc -m "Update the Titanic data"
echo "--- the new pointer:"; cat data/titanic.csv.dvc
echo "--- the remote, one file per version:"; (cd /tmp/dvcstore && find . -type f | sort)

# going back to the earlier version:
git checkout HEAD~1 data/titanic.csv.dvc        # the old pointer, from Git
dvc checkout                                    # the matching data, from the cache
"$PY" -c 'import pandas as pd; df = pd.read_csv("data/titanic.csv"); print("restored:", df.shape, "| alive is back:", "alive" in df.columns)'

rm -rf /tmp/dvcstore                            # remove the demonstration remote
echo "done; the project is left in $WORK"
