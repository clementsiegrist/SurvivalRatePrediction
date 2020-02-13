from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input
from keras.models import Model, Sequential
from keras.layers import Conv1D, GlobalMaxPool1D, LocallyConnected1D, AveragePooling1D, \
    MaxPooling1D, concatenate
import os
import keras
import h5py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from keras.layers import Dense, Dropout
from keras.models import Sequential



def preprocess(path_radiomics, path_clinical_datas, outputs_path, no_output=True):
    '''
    Preprocess the datasets.

    :param path_radiomics: Path where the radiomics datas are stored
    :param path_clinical_datas: Path where the clinical datas are stored
    :param outputs_path: Path where the clinical datas are stored
    :param no_output: If True return a the target datas if False does not return the target datas
    :return: Features, output, and an index of the patient used for post processing
    '''
    radiomics = pd.read_csv(path_radiomics)
    clinical_data = pd.read_csv(path_clinical_datas)
    outputs = pd.read_csv(outputs_path)
    if no_output is True:
        global index_
        index_ = outputs[['PatientID']].values
        outputs = outputs.drop('PatientID', axis=1)
        sv_time = outputs[['SurvivalTime']].to_numpy()
        event = outputs[['Event']].to_numpy()
        outputs = outputs.to_numpy()
        #outputs = np.concatenate((sv_time, event), axis=1)
    else:
        pass

    # Drop useless columns
    clinical_data = clinical_data.drop('Unnamed: 0', axis=1)
    clinical_data = clinical_data.drop([0, 1], axis=0)
    radiomics = radiomics.drop('PatientID', axis=1)
    radio_cate = radiomics[['Histology', 'SourceDataset']]
    radiomics = radiomics.drop(['Histology', 'SourceDataset'], axis=1)
    radio_cate = radio_cate.astype(str).apply(lambda x: x.str.lower()) #map(lambda x: x.lower(), radio_cate)

    # One hot encode columns with string datas
    one_hot = LabelEncoder()
    radio_cate_h = one_hot.fit_transform(radio_cate['Histology'])
    radio_cate_ = one_hot.fit_transform(radio_cate['SourceDataset'])
    radio_cate_h = np.expand_dims(radio_cate_h, 1)
    radio_cate_ = np.expand_dims(radio_cate_, 1)
    radio_cate = np.concatenate((radio_cate_, radio_cate_h), axis=1)
    radio_cate = pd.DataFrame(radio_cate)
    clinical_data.index = radio_cate.index

    # Concat clinical datas and radiomics datas
    radiomics_ = pd.concat([radiomics, radio_cate, clinical_data], axis=1)
    radiomics_ = radiomics_.fillna(value=70).round(3)
    features = radiomics_.to_numpy()
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features = np.expand_dims(features, axis=1)

    # !!! NOT USED FOR PREDICTIONS !!! load images
    images = []
    for i in outputs[['PatientID']].values:
        i = str(i)
        i = i.strip('[]')
        if len(i) == 3:
            print(i)
            archive = os.path.join(path_images, 'patient_' + i + '.npz')
            images.append(archive)
        elif len(i) == 1:  # len(str(outputs.loc[i, 'PatientID'])) == 1:
            archive = os.path.join(path_images, 'patient_' + '00' + i + '.npz')
            images.append(archive)
        else:
            archive = os.path.join(path_images, 'patient_' + '0' + i + '.npz')
            images.append(archive)

    for i in images:
        array = np.load(i)
        scan = array['scan']
        mask = array['mask']

    if no_output is True:
        return features, outputs
    else:
        return features, clinical_data.index



def multi_head_conv1D(trainX, trainy, x_test, epochs, batch_size, testX=None, testy=None, out_put=None, evaluate=False):
    '''
    We use a multihead/multi output dense/conv1D neural network because we can combine different type of layers
    which can be adapted in order to preprocess at the same time heterogeneous datasets (images and/or time series
    and/or categorical variables and or numeric features). As such the newtork is adapted for the clinical and radiomics
    datas but it can be adapted to add the images by modifiyng the shape of the desired inputs as well as the ConvLayers.

    :param trainX: Features of the trainset (radiomics and clinical features)
    :param trainy: Train targets (remaining lifespan and event)
    :param x_test: Features of the testset (radiomics and clinical features)
    :param testX: Features of the evaluation set only if evalute=True
    :param testy: Targets of the evaluation set only if evalute=True
    :param out_put: If set to True use a single dense layer with a sigmoid for multiclass classification
    (cancer / non-cancer / unsure)
    :param evaluate: If set to True one can load an evaluation set
    :param epochs: Number of epochs
    :param batch_size: The number of exemple in a single batch
    :return: List with the predictions
    '''


    n_timesteps, n_features, n_outputs = trainX.shape[2], 1, trainy.shape[1]
    #x_test = np.reshape(x_test.shape[1], 1, x_test.shape[2])

    # head 1 initially for clinal datas
    inputs1 = Input(shape=(n_features, n_timesteps))
    conv1 = Dense(150, activation='elu')(inputs1)
    drop1 = Dropout(0.2)(conv1)
    flat1 = Flatten()(drop1)

    # head 2 initially for radiomics datas
    inputs2 = Input(shape=(n_features, n_timesteps))
    conv2 = Conv1D(filters=128, kernel_size=2, activation='elu', data_format='channels_first')(inputs2)
    drop2 = Dropout(0.2)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)

    # head 3 initially a CONV3D for the images but we had trouble loading the images
    inputs3 = Input(shape=(n_features, n_timesteps))
    conv3 = Conv1D(filters=150, kernel_size=2, activation='elu', data_format='channels_first')(inputs3)
    drop3 = Dropout(0.2)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)
                   
    # merge
    merged = concatenate([flat1, flat2, flat3])

    # interpretation
    dense1 = Dense(100, activation='elu')(merged)

    # In case we have more than two classes to predict
    if out_put == 'multiclass':
        outputs = Dense(n_outputs, activation='softmax')(dense1)
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    else:
        out1 = Dense(1, activation='linear')(dense1) # To approximate the patient remaining lifespan
        out2 = Dense(1, activation='sigmoid')(dense1) # To predict the event (death or survival)
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=[out1, out2])

    # save a plot of the model
    #keras.utils.plot_model(model, show_shapes=True, to_file='multichannel.png')
    from keras.callbacks import ReduceLROnPlateau
    rlrop = ReduceLROnPlateau(monitor='train_loss', factor=0.1, patience=100)
    adam = keras.optimizers.adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

    # fit network
    model.fit([trainX, trainX, trainX], [trainy[:, 0], trainy[:, 1]],
                            epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[rlrop])

    # Make predictions and save the model
    predictions = model.predict([x_test, x_test, x_test])
    model.save('/Users/clementsiegrist/PycharmProjects/App/multi_head.h5')

    # evaluate model
    if evaluate is True:
      _, accuracy = model.evaluate([testX, testX, testX], testy, batch_size=batch_size, verbose=0)
    else:
        return predictions



def compare_predict_test(predictions, threshold):
    '''
    Convert the predictions to 0 is the value is under the threshold and to 1 if the value
    is above the threshold.

    :param predictions: The predicted values
    :param threshold: The threshold you want to set
    :return: An array with the converted values
    '''
    predictions_ = predictions
    print(predictions)

    #  store prediction values in predict
    predict = []
    for i in predictions_:
        if i < threshold:
            predict.append(0)
        elif i > threshold:
            predict.append(1)
        elif i == threshold:
            predict.append(0)

    return predict

if __name__ == '__main__':

        # Path towards datasets
        path_radiomics = '/Users/clementsiegrist/Downloads/data_Q0G7b5t/features/clinical_data.csv'
        path_clinical_datas = '/Users/clementsiegrist/Downloads/data_Q0G7b5t/features/radiomics.csv'
        outputs_path = '/Users/clementsiegrist/Downloads/output_VSVxRFU.csv'
        path_radiomics_test = '/Users/clementsiegrist/Downloads/data_9Cbe5hx/features/clinical_data.csv'
        path_clinical_datas_test = '/Users/clementsiegrist/Downloads/data_9Cbe5hx/features/radiomics.csv'
        path_images = '/Users/clementsiegrist/Downloads/data_Q0G7b5t/images/'
        index = pd.read_csv(path_clinical_datas_test)

        # Preprocess the features and pass them to the neural net
        features, outputs = preprocess(path_radiomics, path_clinical_datas, outputs_path)
        features_, ind = preprocess(path_radiomics_test, path_clinical_datas_test, outputs_path, no_output=False)
        predictions = multi_head_conv1D(features, outputs, features_, batch_size=20, epochs=3000)

        # Post process the predictions and convert them to a csv
        predictions_1 = np.array(predictions[0])
        predictions_2 = np.array(predictions[1])
        predictions_2 = compare_predict_test(predictions_2, 0.5)
        predictions_2 = np.array(predictions_2)
        predictions_2 = np.expand_dims(predictions_2, axis=1)

        data = np.concatenate([predictions_1, predictions_2], axis=1)
        df = pd.DataFrame(data=data, index=ind, columns=['SurvivalTime', 'Event'])
        df.to_csv(path_or_buf='/Users/clementsiegrist/PycharmProjects/App/owkin_results.csv', sep=',')




