
def process_bar(percent, loss, acc, epoch, start_str='', end_str='', total_length=100):
    '''进度条'''
    bar = ''.join(["%s" % '='] * int(percent * total_length)) + ''
    if loss is not None:
        bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent * 100) + end_str + " epoch " + str(epoch) + " loss " + str(loss)
    else:
        bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent*100) + end_str + " epoch " + str(epoch)
    print(bar, end='', flush=True)