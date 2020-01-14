FROM pytorch/pytorch
WORKDIR /workspace

ADD requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir images
RUN mkdir plots
RUN mkdir data

ADD butterfly.png huberROF.py projSimplex.py sparse.py split.py async-master.py ./
CMD python async-master.py
