from argparse import Namespace

HP = Namespace(
    batch_size=128,
    lr=2e-3,
    momentum=0.9,
    lr_decay=0.95,
    optim_type="adam",
    l2_reg=0.0,
    epochs=10,
    do_batchnorm=True,
    p_dropout=0.2
)