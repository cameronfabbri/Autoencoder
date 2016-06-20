#!/bin/bash
for i in *.png; do echo "<img src='$i' />" >> index.html; done;
python -m SimpleHTTPServer
