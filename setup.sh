#!/bin/bash

# Copy HW data to the user home folder
rm -d lost+found && cp -r /homework/* /home/jovyan/

# Execute populating of atomic data
python populate.py

cd /home/jovyan/modules
chmod 755 *.py
chmod 755 /home/jovyan/modules