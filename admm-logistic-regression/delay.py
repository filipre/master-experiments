"""

"""
import random

def forMaster(xk_models_queue, maxDelay=1, method='nodelay'):
    xk_models = []
    nodes = len(xk_models_queue)

    # no delay
    for k in range(nodes):
        xk_model = xk_models_queue[k][0]
        xk_models.append(xk_model)

    if method == 'constant':
        for k in range(nodes):
            xk_model = _constantDelay(xk_models_queue[k])
            xk_models[k] = xk_model

    elif method == 'uniform':
        for k in range(nodes):
            xk_model = _uniformDelay(xk_models_queue[k])
            xk_models[k] = xk_model

    # elif args.delay_type == 'mixture':
    #     for k in range(args.nodes // 2):
    #         delays.append(0)
    #     for k in range(args.nodes//2, args.nodes):
    #         delay = constantDelay.choose(uk_delays[k])
    #         delays.append(delay)
    #
    # elif args.delay_type == 'oneway':
    #     for k in range(args.nodes):
    #         delay = constantDelay.choose(uk_delays[k])
    #         delays.append(delay)
    #
    # elif args.delay_type == 'warmup':
    #     for k in range(args.nodes):
    #         delay = 0 if t < args.delay else args.delay - 1
    #         delays.append(delay)

    return xk_models

def forWorker(x0_model_queue, maxDelay=1, method='nodelay'):

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
