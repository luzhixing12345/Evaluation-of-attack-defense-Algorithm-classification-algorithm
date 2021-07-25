import matplotlib.pyplot as plt
import os


def evaluate(cfg,train_losses,train_counter,test_accurency,test_counter):
    evaluate_train_loss(cfg,train_losses,train_counter)
    evaluate_accurency(cfg,test_accurency,test_counter)



def evaluate_train_loss(cfg,train_losses,train_counter):
    plt.plot(train_counter, train_losses, color='blue')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.savefig('./train&test/'+cfg.dataset+'_'+cfg.network+'_train_loss.jpg')
    plt.close()
    #plt.show()
    

def evaluate_accurency(cfg,test_accurency,test_counter):
    plt.plot(test_counter,test_accurency,color='blue')
    plt.xlabel('epoch')
    plt.ylabel('accurency')
    plt.savefig('./train&test/'+cfg.dataset+'_'+cfg.network+'_test_accurency.jpg')
    plt.close()
    #plt.show()

def show_pic(example_data,example_targets):
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()
