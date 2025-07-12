#!/bin/bash
set -e

echo "[Demo] Initializing demo repository..."
rm -rf demo/conflict_scenarios/demo-repo
mkdir -p demo/conflict_scenarios/demo-repo
cd demo/conflict_scenarios/demo-repo
git init

echo "First file" > file.txt
git add file.txt
git commit -m "Initial commit"

git checkout -b feature/alice

echo "Alice's change" >> file.txt
git commit -am "Alice edit"

git checkout main
git checkout -b feature/bob

echo "Bob's change" >> file.txt
git commit -am "Bob edit"

git checkout main

echo "[Demo] Created branches: feature/alice, feature/bob with conflicting changes in file.txt."
echo "[Demo] To test, run:"
echo "  python ../../../src/main.py predict"
echo "  python ../../../src/main.py detect"
echo "  python ../../../src/main.py dashboard" 