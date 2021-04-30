def model_setup(ipt_dim, NN_dic):
    #import useful packages
    import keras
    import pandas as pd
    from keras.models import Sequential
    from keras.layers import Dense, Dropout

    import numpy as np
    import keras.backend as kb
    import tensorflow as tf

    import matplotlib

    model = Sequential()
    
    for idx in NN_dic.index:
        if idx==0:
            model.add(Dense(NN_dic.loc[idx,'Number_of_neurons'], activation=NN_dic.loc[idx,'Activation_Func'], input_dim=ipt_dim))
        else:
            model.add(Dense(NN_dic.loc[idx,'Number_of_neurons'], activation=NN_dic.loc[idx,'Activation_Func']))
        model.add(Dropout(NN_dic.loc[idx,'Dropout']))

    model.summary()
    return model

def model_train_cell(model, x_train, x_test, y_train, y_test, learning_rate, epoch, earlystop_patience):
    import pandas as pd
    import numpy as np
    import keras
    from keras import backend as K

    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.wrappers.scikit_learn import KerasRegressor
    
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split

    import matplotlib.pyplot as plt

    #from tf.keras import optimizers
    rms = keras.optimizers.RMSprop(learning_rate)
    model.compile(loss='mean_absolute_error',optimizer=rms)

    # Add an early stopping callback
    es = keras.callbacks.EarlyStopping(
        monitor='loss', 
        mode='min', 
        patience = earlystop_patience, 
        restore_best_weights = True, 
        verbose=1)
    # Add a checkpoint where loss is minimum, and save that model
    mc = keras.callbacks.ModelCheckpoint('best_model.SB', monitor='loss', 
                        mode='min',  verbose=1, save_best_only=True)

    #historyData = model.fit(xarray,yarray,epochs=800,callbacks=[es])
    historyData = model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=epoch, batch_size=10)

    trn_loss_hist = historyData.history['loss']
    val_loss_hist = historyData.history['val_loss']
    #epoch_hist = historyData.epoch

    #The above line will return a dictionary, access it's info like this:
    min_trn_loss_index = np.argmin(trn_loss_hist)
    best_epoch = min_trn_loss_index + 1
    print ('best epoch = ', best_epoch)
    print('smallest training loss =', np.min(trn_loss_hist))
    print('corresponding validation loss = ',  val_loss_hist[min_trn_loss_index] )
    # print(trn_loss_hist)
    
    return trn_loss_hist, val_loss_hist

def model_train(model, x_train, x_test, y_train, y_test, Training_dic):
    import numpy as np
    import matplotlib.pyplot as plt
    
    val_loss_hist = []
    trn_loss_hist = []
    epoch_total = 0
    for i in Training_dic.index:
        trn, val = model_train_cell(model, x_train, x_test, y_train, y_test, Training_dic.loc[i,'Learning_rate'], Training_dic.loc[i,'Epochs'], Training_dic.loc[i,'Earlystop_patience'])
        
        val_loss_hist = val_loss_hist + val
        trn_loss_hist = trn_loss_hist + trn
        epoch_total = epoch_total + Training_dic.loc[i,'Epochs']
    
    epoch_hist = range(epoch_total)

    plt.figure(num=1, figsize=(8, 5))
    plt.plot(epoch_hist, trn_loss_hist, 'k-')    
    plt.plot(epoch_hist, val_loss_hist, 'r-')    

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['training set', 'validation set'])

    plt.show(block=False)
    plt.pause(0.05)
    plt.close()

    return model
    
def model_check(model, x_trn, x_vad, y_trn, y_vad, X_med, Y_med):
    #import math
    import numpy as np
    import keras
    import keras.backend as kb
    import tensorflow as tf

    import matplotlib.pyplot as plt

    # predict data using Keras model
    # first point (row [0])comparison of data and prediction
    # put in a loop to print comparion for all data points
    model_op = []
    for i in range(len(x_trn)):
        model_ip = [[ x_trn[i][0] , x_trn[i][1] , x_trn[i][2] ]]
        model_ip_array = np.array(model_ip)
        model_op_array = model.predict(model_ip_array)
        model_op.append(model_op_array)

    Y_pred_trn = np.array(model_op)
    Y_pred_trn = np.squeeze(Y_pred_trn)

    Y_real_trn = y_trn


    model_op = []
    for i in range(len(x_vad)):
        model_ip = [[ x_vad[i][0] , x_vad[i][1] , x_vad[i][2] ]]
        model_ip_array = np.array(model_ip)
        model_op_array = model.predict(model_ip_array)
        model_op.append(model_op_array)

    Y_pred_vad = np.array(model_op)
    Y_pred_vad = np.squeeze(Y_pred_vad)

    Y_real_vad = y_vad

    for yy in range(len(Y_med)):
        Y_pred_trn[:,yy] = Y_pred_trn[:,yy]*Y_med[yy]
        Y_real_trn[:,yy] = Y_real_trn[:,yy]*Y_med[yy]
        Y_pred_vad[:,yy] = Y_pred_vad[:,yy]*Y_med[yy]
        Y_real_vad[:,yy] = Y_real_vad[:,yy]*Y_med[yy]

        plt.figure(num=yy+2, figsize=(8, 5))
        plt.plot(Y_pred_trn[:,yy], Y_real_trn[:,yy], 'k.')
        plt.plot(Y_pred_vad[:,yy], Y_real_vad[:,yy], 'r.')
        plt.legend(['training set', 'validation set'])
        plt.xlabel('predicted output using training dataset')
        plt.ylabel('real output in training dataset')
        plt.show()
    # plt.pause(0.05)
    # plt.close()    

    print('Y_pred_trn', Y_pred_trn)
    print('Y_real_trn', Y_real_trn)
    print('Y_pred_vad', Y_pred_vad)
    print('Y_real_vad', Y_real_vad)


