#!/bin/sh
for d in */ ; do
  path=$(realpath "$d")
  cd ..
  cd "DNSMOS"
  python "C:\Dev\deeplearning\dl-speech-enhancement\AudioDec\DNSMOS\dnsmos_local.py" -t "$path" -o "$path/output.csv"
  cd ..
  cd "test_out"
done