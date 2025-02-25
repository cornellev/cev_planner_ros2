#!/bin/bash

cd cev_planner && git add . && git commit -m "$1" && cd ..
git add . && git commit -m "$1"
