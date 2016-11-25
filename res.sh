#!/bin/bash
for f in images/*.png; do
   convert "$f" -resize 200x200 "$f"
done
