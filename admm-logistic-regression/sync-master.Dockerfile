FROM pytorch/pytorch
WORKDIR /workspace

ADD requirements.txt .
RUN pip install -r requirements.txt

ADD MNIST MNIST
RUN mkdir graphs

ADD dataloader.py model.py x0SolverNoMult.py x0SolverWithMult.py augLagrangianNoMult.py augLagrangianWithMult.py sync-master.py ./
CMD python sync-master.py
