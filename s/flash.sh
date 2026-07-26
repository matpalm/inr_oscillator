#!/usr/bin/env bash
set -ex
cd ~/dev/tiliqua/gateware
openFPGALoader -c dirtyJtag /home/mat/dev/inr_waveshaper/runs/$1/tiliqua/top.bit