#!/bin/sh
cd data
find * -type d -exec tar -cz {} -f {}.tar.gz \;
cd ..
