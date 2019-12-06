FROM pytorch/pytorch
WORKDIR /workspace

ADD requirements.txt .
RUN pip install -r requirements.txt

ADD MNIST MNIST

ADD dataloader.py model.py xkSolverNoMult.py xkSolverWithMult.py ykSolver.py async-worker.py ./
CMD python async-worker.py
