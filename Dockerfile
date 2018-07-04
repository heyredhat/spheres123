FROM heroku/miniconda

# Grab requirements.txt.
ADD ./spheres/requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install -qr /tmp/requirements.txt

# Add our code
ADD ./spheres /opt/spheres/
WORKDIR /opt/spheres

RUN conda install qutip

CMD gunicorn --bind 0.0.0.0:$PORT wsgi