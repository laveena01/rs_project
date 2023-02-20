import matplotlib.pyplot as plt
import numpy as np

def show(epoch, TP, FP, Precision):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xticks(epoch)
    ax.plot(epoch, TP, '-b', label='TP')
    ax.plot(epoch, FP, '-r', label='FP')
    ax2 = ax.twinx()
    ax2.plot(epoch, Precision, '-g', label='Precision')
    fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)

    ax.set_xlabel('epoch')
    ax.set_ylabel(r'TP')
    ax2.set_ylabel(r'Precision')
    plt.show()
if __name__ == "__main__":
    x = []
    for i in range(55):
        x.append(i)
    y1 = np.random.randint(220, 300, 55)
    y2 = np.random.randint(30, 150, 55)
    y3 = []
    for i in range(55):
        y3.append(float(y1[i] + y2[i]) / y1[i])
    show(x, y1, y2, y3)