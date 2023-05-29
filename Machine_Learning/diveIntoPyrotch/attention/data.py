import torch
from torch import nn
import matplotlib.pyplot as plt


def f(x):
    return 2 * torch.sin(x) + x**0.8


def create_diy_data(n_train = 50):
    x_train, _ = torch.sort(torch.rand(n_train) * 5)   # 排序后的训练样本
    y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
    x_test = torch.arange(0, 5, 0.1)  # 测试样本
    y_truth = f(x_test)  # 测试样本的真实输出
    return x_train, y_train, x_test, y_truth

def plot_kernel_reg(x_train, y_train, x_test, y_truth, y_hat):
    # plt.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
    #          xlim=[0, 5], ylim=[-1, 5])
    plt.plot(x_test, y_truth)
    plt.plot(x_test, y_hat)
    plt.plot(x_train, y_train, 'o', alpha=0.5)
    plt.show()

#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """显示矩阵热图"""
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    plt.show()

if __name__ == "__main__":
    n_train = 50
    x_train, y_train, x_test, y_truth = create_diy_data(n_train=n_train)
    y_hat = torch.repeat_interleave(y_train.mean(), len(x_test))
    print(x_test.shape, y_truth.shape, y_hat.shape)
    # plot_kernel_reg(x_train, y_train, x_test, y_truth, y_hat)

    # X_repeat的形状:(n_test,n_train),
    # 每一行都包含着相同的测试输入（例如：同样的查询）
    X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
    # x_train包含着键。attention_weights的形状：(n_test,n_train),
    # 每一行都包含着要在给定的每个查询的值（y_train）之间分配的注意力权重
    attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
    # y_hat的每个元素都是值的加权平均值，其中的权重是注意力权重
    y_hat = torch.matmul(attention_weights, y_train)
    # plot_kernel_reg(x_train, y_train, x_test, y_truth, y_hat)
    show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')