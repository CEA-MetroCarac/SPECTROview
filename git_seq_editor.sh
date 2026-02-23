#!/bin/bash
sed -i.bak -E 's/^pick (09807f1.*)/edit \1/' "$1"
