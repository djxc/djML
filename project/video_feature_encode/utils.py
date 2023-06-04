import matplotlib.pyplot as plt

def show_loss(log_file_path):
    train_loss_list = []
    val_loss_list = []
    epoch_train_list = []
    epoch_val_list = []
    val_acc_list = []

    with open(log_file_path) as log_file:
        log_str = log_file.read()
        log_str = log_str.split("start training......")[-1]
        log_str_lines = log_str.split("\n")
        epoch = 0
        for i, log_line in enumerate(log_str_lines):
            if log_line.startswith("epoch"):
                log_lines = log_line.split(", ")
                epoch = int(log_lines[0].split(" ")[1])
                epoch_train_list.append(epoch)
                train_loss_str = log_lines[1].replace("loss ", "")
                train_loss_list.append(float(train_loss_str))

            elif log_line.startswith("train acc"):
                epoch_val_list.append(epoch)
                val_acc_loss = log_line.split(";")[0].split(",")

                val_acc_list.append(float(val_acc_loss[0].split(": ")[1]))
                val_loss_list.append(float(val_acc_loss[1].split(": ")[1]))

    plt.plot(epoch_train_list, train_loss_list, color="red")
    plt.plot(epoch_val_list, val_loss_list, color="orange")
    plt.plot(epoch_val_list, val_acc_list, color="green")

    plt.show()
   

if __name__ == "__main__":
    show_loss(r"D:\Data\MLData\videoFeature\train_log_leNet_bn_pre.txt")