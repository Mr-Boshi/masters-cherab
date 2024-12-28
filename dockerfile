# Use the official Jupyter datascience-notebook as a base image
FROM jupyter/datascience-notebook:latest

# Switch to root to install additional packages and create directories
USER root

# Install the CHERAB package from PyPI using pip and create required directories in one RUN command
RUN mkdir -p /homework/modules /homework/data && \
    pip install cherab

# Copy the Python script and other resources to the container
COPY HW_cherab.ipynb /homework/HW_cherab.ipynb
COPY modules/ /homework/modules/
COPY data/ /homework/data/
COPY populate.py /homework/populate.py
COPY setup.sh /homework/setup.sh

RUN chown -R $NB_UID:$NB_GID /homework

# Switch back to the default notebook user
USER $NB_UID

# Expose default Jupyter port
EXPOSE 8888

# Set default command for Jupyter start
CMD ["start-singleuser.sh"]