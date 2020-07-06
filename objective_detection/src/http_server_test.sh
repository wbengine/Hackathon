#!/usr/bin/env bash


curl -v http://127.0.0.1:8080/ -F name=dhellmann -F foo=bar \
-F image=@/Users/lls/Works/Hackathon2019/objective_detection/images/panda.jpg
