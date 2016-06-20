#!/bin/bash
> index.html
for i in *.png; do echo "<img src='$i' /><br><br>" >> index.html; done;
python -m SimpleHTTPServer
