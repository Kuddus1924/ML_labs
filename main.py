import matplotlib.pyplot as plt
import numpy as np

import trainer
import linreg
import logreg

if __name__ == '__main__':
    trainer.predict_via_iterations(
        'linreg/t1_linreg_x_train.csv',
        'linreg/t1_linreg_y_train.csv',
        'linreg/t1_linreg_x_test.csv',
        'linreg/t1_linreg_y_test.csv',
        linreg.LinregGradientDescent,
        1e-4,
        int(1e5),
        0.7
    )

    trainer.predict_via_iterations(
        'logreg/t1_logreg_x_train.csv',
        'logreg/t1_logreg_y_train.csv',
        'logreg/t1_logreg_x_test.csv',
        'logreg/t1_logreg_y_test.csv',
        logreg.LogregGradientDescent,
        5e-2,
        int(1e4),
        0.7
    )

    plt.show()
