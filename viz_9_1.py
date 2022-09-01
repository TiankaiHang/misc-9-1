# ref: https://blog.csdn.net/bingningning/article/details/79807345


import numpy as np
import matplotlib.pyplot as plt


# color setting
# https://cloud.tencent.com/developer/article/1673268
COLOR_LIST = [ 
    'xkcd:green',
    'xkcd:pink',
    'xkcd:blue',
    'xkcd:violet',
]

DATA = [ 
    [ 
        [0, 20, 40],
        [82.36, 72.28, 59.8],
        [83.56, 75.76, 64.64],
        [83.8, 76.72, 66.8],
    ],
    [ 
        [0, 20, 40],
        [82.44, 72.16, 60.88],
        [83.52, 74.48, 62.36],
        [84.24, 75.16, 64.3]
    ]
]

DATA = np.array(DATA)

legend = [ 
    "scratch", "orig-learngene", 
    "meta-learngene"
]

WIDTH = 60

for i in range(len(DATA)):
    _data = DATA[i]
    # print(_data)

    # for j in range(_data.shape[1]):
    #     plt.bar(
    #         x=_data[0][j] + 3 * np.array([0, 30, 60]),
    #         height=_data[1:, j],
    #         width=WIDTH / (len(_data) - 1),
    #         label=legend[j],
    #         fc=COLOR_LIST[j],
    #     )

    for j in range(3):
        _x = _data[0][j] + 3 * np.array([0, 30, 60])
        _y = _data[1 + j]
        plt.bar(
            x=_x,
            height=_y,
            width=WIDTH / (len(_data) - 1),
            label=legend[j],
            fc=COLOR_LIST[j],
        )

        # add number
        for a, b in zip(_x, _y):
            plt.text(
                a, b,
                b,
                ha='center', 
                va='bottom',
                fontsize=8,
            )


    plt.xticks(np.array([0, 90, 180]) + 20, ["0", "20", "40"])
    plt.ylim(0, 100)

    plt.legend()
    plt.savefig(f'{i}.png')
    plt.clf()
