#!/bin/bash
if [[ "$1" == *git-rebase-todo ]]; then
    sed -i.bak -E 's/^pick (09807f1.*)/r \1/' "$1"
else
    echo "#10 Optimize save_work() and load_work() methode. And Fix bug related to save and load fit mode" > "$1"
fi
