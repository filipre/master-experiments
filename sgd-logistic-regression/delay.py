import random

def delayModel(x0_model_queue, maxDelay=1, method='nodelay'):

    # no delay
    x0_model = x0_model_queue[0]

    if method == 'constant':
        x0_model = _constantDelay(x0_model_queue)

    elif method == 'uniform':
        x0_model = _uniformDelay(x0_model_queue)

    return x0_model


def _constantDelay(model_queue):
    delay = len(model_queue) - 1
    return model_queue[delay]

def _uniformDelay(model_queue):
    delay = random.randrange(0, len(model_queue))
    return model_queue[delay]
