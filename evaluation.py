import matplotlib.pyplot as plt




def evaluate_train_loss(train_losses,train_counter):
    plt.plot(train_counter, train_losses, color='blue')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.savefig('train_loss.jpg')
    plt.show()
    

def evaluate_accurency(test_accurency,test_counter):
    plt.plot(test_accurency,test_counter, color='blue')
    plt.xlabel('epoch')
    plt.ylabel('accurency')
    plt.savefig('test_accurency.jpg')
    plt.show()

def show_pic(example_data,example_targets):
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()
    