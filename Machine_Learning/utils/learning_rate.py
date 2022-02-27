def warm_up_lr(dk_num, step, warm_up_steps):
    """
    Warm-up learning rate
    :param step: current step
    :param warm_up_steps: warm-up steps
    :param init_lr: initial learning rate
    :return: learning rate
    """
    lr = dk_num ** 0.5 * min(step ** -0.5, step * warm_up_steps ** -1.5)
    return lr
