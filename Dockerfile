FROM heroku/miniconda

# Grab requirements.txt.
ADD ./webapp/requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install -qr /tmp/requirements.txt

# Add our code
ADD ./webapp /opt/webapp/
WORKDIR /opt/webapp

RUN conda install sympy
RUN conda install cython
RUN conda install -c conda-forge qutip=4.2 
RUN conda install libgcc

CMD gunicorn --log-level debug -k eventlet -w 1 --bind 0.0.0.0:$PORT wsgi