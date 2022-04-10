# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 21:21:02 2022

@author: 81916
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 18:59:50 2022

@author: 81916
"""
# -*- coding:utf-8
import numpy as np
import struct
import pickle
class Data:
    def __init__(self, stepsize, hiden_layer, reg_factor, train_img_list, train_label_list,
                            test_img_list, test_label_list):

        self.K = 10
        self.N = 60000
        self.M = 10000
        self.BATCHSIZE = 2000
        self.reg_factor = reg_factor#1e-3
        self.original_stepsize = stepsize
        self.stepsize = stepsize#1e-2
        self.train_img_list = train_img_list
        self.train_label_list = train_label_list

        self.test_img_list = test_img_list
        self.test_label_list = test_label_list

        self.train_loss_list = []
        self.test_loss_list = []
        self.test_accuracy_list = []
        
        self.hiden_layer =  hiden_layer#100
        self.init_network()

        self.train_data = np.append( self.train_img_list, self.train_label_list, axis = 1 )
        
        self.maxEpochs = 1000
        

    def predict(self):
        hidden_layer1 = np.maximum(0, np.matmul(self.test_img_list, self.W1) + self.b1)


        #hidden_layer2 = np.maximum(0, np.matmul(hidden_layer1, self.W2) + self.b2)
        scores = np.maximum(0, np.matmul(hidden_layer1, self.W2) + self.b2)

        #scores = np.maximum(0, np.matmul(hidden_layer2, self.W3) + self.b3)

        prediction = np.argmax( scores, axis = 1 )
        prediction = np.reshape( prediction, ( 10000,1 ) )
        
        accuracy = np.mean( prediction == self.test_label_list )
        print ('Accuracy: {}'.format(accuracy))
        return accuracy

    def train(self):

        for i in range(self.maxEpochs):#lr decay
            if i % 200 == 0:
                self.stepsize = self.stepsize * 0.5
            
            np.random.shuffle( self.train_data )
            img_list= self.train_data[:self.BATCHSIZE,:-1]
            label_list = self.train_data[:self.BATCHSIZE, -1:]
            #print ("Train Time: ",i)
            self.train_network(img_list, label_list)
            self.predict_network()

        
    def train_network(self, img_batch_list, label_batch_list):
        # calculate softmax
        train_example_num = img_batch_list.shape[0]
        hidden_layer1 = np.maximum( 0, np.matmul( img_batch_list, self.W1 ) + self.b1 )

        #hidden_layer2 = np.maximum( 0, np.matmul( hidden_layer1, self.W2 ) + self.b2 )

        scores = np.maximum(0, np.matmul(hidden_layer1, self.W2) + self.b2)

        scores_e = np.exp( scores )
        
        scores_e_sum = np.sum( scores_e, axis = 1, keepdims= True )

        probs = scores_e / scores_e_sum
        #compute loss
        loss_list_tmp = np.zeros( (train_example_num, 1) )
        for i in range( train_example_num ):
            loss_list_tmp[ i ] = scores_e[ i ][ int(label_batch_list[ i ]) ] / scores_e_sum[ i ]
        
        loss_list = -np.log( loss_list_tmp )
        
        loss = np.mean( loss_list, axis=0 )[0] + \
                0.5 * self.reg_factor * np.sum( self.W1 * self.W1 ) + \
                0.5 * self.reg_factor * np.sum( self.W2 * self.W2 ) #+ \
                #0.5 * self.reg_factor * np.sum( self.W3 * self.W3 )

        self.train_loss_list.append(loss)
        if (len(self.train_loss_list) % 100 == 0):
            print (loss, " ", len(self.train_loss_list))
        
        # backpropagation

        dscore = np.zeros( (train_example_num, self.K) )
        
        for i in range( train_example_num ):
            dscore[ i ][ : ] = probs[ i ][ : ]
            dscore[ i ][ int(label_batch_list[ i ]) ] -= 1

        dscore /= train_example_num


        dW2 = np.dot( hidden_layer1.T, dscore )
        db2 = np.sum( dscore, axis = 0, keepdims= True )
        
        dh1 = np.dot(dscore, self.W2.T )
        dh1[ hidden_layer1 <= 0 ] = 0

        dW1 = np.dot( img_batch_list.T, dh1 )
        db1 = np.sum( dh1, axis = 0, keepdims= True )



        #dW3 += self.reg_factor * self.W3
        dW2 += self.reg_factor * self.W2
        dW1 += self.reg_factor * self.W1


        #self.W3 += -self.stepsize * dW3
        self.W2 += -self.stepsize * dW2
        self.W1 += -self.stepsize * dW1

        #self.b3 += -self.stepsize * db3
        self.b2 += -self.stepsize * db2
        self.b1 += -self.stepsize * db1


        return
    def predict_network(self):
        hidden_layer1 = np.maximum(0, np.matmul(self.test_img_list, self.W1) + self.b1)


        #hidden_layer2 = np.maximum(0, np.matmul(hidden_layer1, self.W2) + self.b2)
        scores = np.maximum(0, np.matmul(hidden_layer1, self.W2) + self.b2)

        #scores = np.maximum(0, np.matmul(hidden_layer2, self.W3) + self.b3)
        scores_e = np.exp( scores )
        
        scores_e_sum = np.sum( scores_e, axis = 1, keepdims= True )

        loss_list_tmp = np.zeros( (self.M, 1) )
        for i in range( self.M ):
            loss_list_tmp[ i ] = scores_e[ i ][ int(self.test_label_list[ i ]) ] / scores_e_sum[ i ]
        
        loss_list = -np.log( loss_list_tmp )
        
        loss = np.mean( loss_list, axis=0 )[0] + \
                0.5 * self.reg_factor * np.sum( self.W1 * self.W1 ) + \
                0.5 * self.reg_factor * np.sum( self.W2 * self.W2 ) #+ \
                #0.5 * self.reg_factor * np.sum( self.W3 * self.W3 )

        self.test_loss_list.append(loss)

        prediction = np.argmax( scores, axis = 1 )
        prediction = np.reshape( prediction, ( 10000,1 ) )
        #print (prediction.shape)
        #print (self.test_label_list.shape)
        accuracy = np.mean( prediction == self.test_label_list )
        #print (accuracy)
        self.test_accuracy_list.append(accuracy)
        return
       

    def init_network(self):
        self.W1 = 0.01 * np.random.randn( 28 * 28, self.hiden_layer )
        self.b1 = 0.01 * np.random.randn( 1, self.hiden_layer )

        self.W2 = 0.01 * np.random.randn(self.hiden_layer, self.K )
        self.b2 = 0.01 * np.random.randn( 1, self.K )

    

def read_train_images(filename):
    train_img_list = np.zeros((60000, 28 * 28))   
    binfile = open(filename, 'rb')
    buf = binfile.read()
    binfile.close()
    index = 0
    magic, train_img_num, numRows, numColums = struct.unpack_from('>IIII', buf, index)
    #print magic, ' ', self.train_img_num, ' ', self.numRows, ' ', self.numColums
    index += struct.calcsize('>IIII')
    for i in range(train_img_num):
        im = struct.unpack_from('>784B', buf, index)
        index += struct.calcsize('>784B')
        im = np.array(im)
        im = im.reshape(1, 28 * 28)
        train_img_list[ i , : ] = im
    return    train_img_list
            # plt.imshow(im, cmap='binary')  # 黑白显示
            # plt.show()

def read_train_labels(filename):
    binfile = open(filename, 'rb')
    index = 0
    buf = binfile.read()
    binfile.close()

    magic, train_label_num = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')
    train_label_list = np.zeros((60000, 1))
    for i in range(train_label_num):
        # for x in xrange(2000):
        label_item = int(struct.unpack_from('>B', buf, index)[0])
        train_label_list[ i , : ] = label_item
        index += struct.calcsize('>B')
    return train_label_list

def read_test_images(filename):
    binfile = open(filename, 'rb')
    buf = binfile.read()
    binfile.close()
    index = 0
    test_img_list = np.zeros((10000, 28 * 28))
    magic, test_img_num, numRows, numColums = struct.unpack_from('>IIII', buf, index)
    #print magic, ' ', self.test_img_num, ' ', self.numRows, ' ', self.numColums
    index += struct.calcsize('>IIII')
    for i in range(test_img_num):
        im = struct.unpack_from('>784B', buf, index)
        index += struct.calcsize('>784B')
        im = np.array(im)
        im = im.reshape(1, 28 * 28)
        test_img_list[i, :] = im
    return test_img_list

def read_test_labels(filename):
    binfile = open(filename, 'rb')
    index = 0
    buf = binfile.read()
    binfile.close()
    test_label_list = np.zeros((10000, 1))
    magic, test_label_num = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')

    for i in range(test_label_num):
        # for x in xrange(2000):
        label_item = int(struct.unpack_from('>B', buf, index)[0])
        test_label_list[i, :] = label_item
        index += struct.calcsize('>B')
    return test_label_list                 
    

if __name__ == '__main__':
    train_img_list = read_train_images( '.\\data\\train-images.idx3-ubyte')
    train_label_list = read_train_labels( '.\\data\\train-labels.idx1-ubyte')
    test_img_list = read_test_images('.\\data\\t10k-images.idx3-ubyte')
    test_label_list = read_test_labels('.\\data\\t10k-labels.idx1-ubyte')
    stepsize_list = [1e-2, .5e-2, 1e-3]
    hidden_list = [100, 150, 200]
    reg_factor_list = [1e-2, 1e-3, .5e-2]
    max_acc = 0
    acc_list = []
    #grid search
    for stepsize in stepsize_list:
        for hidden in hidden_list:
            for reg_factor in reg_factor_list:
                data = Data(stepsize, hidden, reg_factor, train_img_list, train_label_list,
                            test_img_list, test_label_list)
                data.train()
                pre = data.predict()
                acc_list.append({'step':stepsize, 'hide':hidden, 'reg':reg_factor,'acc':pre})
                if pre > max_acc:
                     max_acc = pre
                     final_model = data
    with open("network.model", "wb" ) as f:
        f.write(pickle.dumps(final_model))
    f.close()
    


