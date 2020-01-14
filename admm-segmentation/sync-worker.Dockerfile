FROM pytorch/pytorch
WORKDIR /workspace

ADD requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir images
RUN mkdir plots

ADD butterfly.png huberROF.py projSimplex.py sparse.py split.py sync-worker.py ./
CMD python sync-worker.py
