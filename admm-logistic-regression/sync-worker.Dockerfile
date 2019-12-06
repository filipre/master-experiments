FROM pytorch/pytorch
WORKDIR /workspace

ADD requirements.txt .
RUN pip install -r requirements.txt

ADD MNIST MNIST

ADD dataloader.py model.py xkSolverNoMult.py xkSolverWithMult.py ykSolver.py sync-worker.py ./
CMD python sync-worker.py
