# Use the official Jupyter datascience-notebook as a base image
FROM jupyter/datascience-notebook:latest

# Switch to root to install additional packages
USER root

# Install necessary system dependencies for CHERAB, if any
# RUN apt-get update && apt-get install -y \
#     python3-dev \
#     libfftw3-dev \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# Switch back to the default notebook user
USER $NB_UID

# Install the CHERAB package from PyPI using pip
RUN pip install cherab

# Copy the Python script to the container
COPY populate.py ./

# Run the Python script
RUN python populate.py

# Delete the Python script after execution
RUN rm -f ./populate.py

USER root
RUN mkdir -p /homework/modules
RUN mkdir -p /homework/data
COPY HW_cherab.ipynb /homework/HW_cherab.ipynb
COPY modules/* /homework/modules/
COPY data/* /homework/data/

RUN chown -R 1000:100 /homework
USER $NB_UID

# Expose default Jupyter port
EXPOSE 8888

# Set default command for Jupyter start
CMD ["start-singleuser.sh"]
