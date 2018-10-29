#!/bin/sh
cd data
find -name "*.tar.gz" -exec tar -xzf {} \;
cd ..
