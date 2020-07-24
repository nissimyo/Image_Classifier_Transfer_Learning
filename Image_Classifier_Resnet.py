from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from keras.applications.resnet_v2 import ResNet50V2, preprocess_input, decode_predictions
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.models import Model, Sequential
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.io
import time
import cv2
import os


# <editor-fold desc="Data preprocessing">
def GetDefaultParameters():
    '''
        initiating the pipe parameters dictionary
    '''

    path = 'C:/Users/idofi/OneDrive/Documents/BGU\masters/year A/computer vision/Task_2/FlowerData'
    test_images_indices = list(range(301, 473))
    image_size = 224
    threshold = 0.5
    dataAugment = 1
    batch_size = 32
    epochs = 3
    SGD_learning_rate = 0.7
    SGD_momentum = 0.1
    SGD_momentum = 0.2
    SGD_nesterov = False

    params = {'Data': {'path': path, 'test_images_indices': test_images_indices},
              'Prepare': {'size': image_size, 'dataAugment': dataAugment},
              'Model': {'batch_size': batch_size, 'epochs': epochs, 'threshold': threshold},
              'Optimizer': {'SGD': {'learning_rate': SGD_learning_rate, 'momentum': SGD_momentum,
                                    'nesterov': SGD_nesterov}}}

    return params


def GetData(Params, saveToPkl):
    '''
    creating dataframe with the images & labels data
    :param  Params: for the folder path
    :param  saveToPkl: serializing to pickle file (boolean)
    :return dataframe
    '''

    folder_path = Params['Data']['path']

    print(' ---- Importing data ---- ')
    mat_file = scipy.io.loadmat('{}/FlowerDataLabels.mat'.format(folder_path))
    labels = mat_file['Labels'][0]
    raw_data = pd.DataFrame(columns=['Img_ID', 'Data', 'Labels'])

    # read images from folder
    folder_files = os.listdir(folder_path)
    for file in folder_files:
        if file.endswith(('.jpeg')):
            Img_ID = int(file.replace('.jpeg', ''))
            readed_image = cv2.imread('{}\{}'.format(folder_path, file))
            image_label = labels[Img_ID - 1]
            raw_data = raw_data.append({'Img_ID': Img_ID, 'Data': readed_image, 'Labels': image_label},
                                       ignore_index=True)

    # sort daraframe by img_ID
    sorted_raw_data = raw_data.sort_values(['Img_ID'])
    sorted_raw_data = sorted_raw_data.reset_index(drop=True)

    if saveToPkl:
        sorted_raw_data.to_pickle('raw_data.pkl')

    return sorted_raw_data


def TrainTestSplit(Params, DandL):
    '''
    splits the data into train and test
    :param DandL: raw dataframe
    :param Params
    :return: TrainData & TestData (in dataframe format)
    '''

    test_images_indices = Params['Data']['test_images_indices']

    TestData = DandL[DandL['Img_ID'].isin(test_images_indices)]
    TrainData = DandL[~DandL['Img_ID'].isin(test_images_indices)]

    return TrainData, TestData


def prepare(Params, data):
    '''
       resizing the images to fit the resNet50V2
       performig preprocessing and fit it into numpy array
       :param:data: image + labels dataframe
       :param: Params: pipe parameters
       :return: dataframe with resized images
       '''

    copy_data = data.copy()
    size = Params['Prepare']['size']
    # resizing the images
    copy_data['Data'] = copy_data['Data'].apply(lambda x: cv2.resize(x, (size, size)))

    # applying resNetV2 pre_processing to fit the model
    copy_data['Data'] = copy_data['Data'].apply(lambda x: preprocess_input(x))

    train_Set = []
    labels_set = []

    for img in copy_data['Data']:
        train_Set.append(img)

    for label in copy_data['Labels']:
        labels_set.append(label)

    train_Set = np.array(train_Set)
    labels_set = np.array(labels_set)

    return train_Set, labels_set



# <editor-fold desc="Baseline model">
def build_baseline_model(Params):
    '''
    bulidng a pre-trined baseline model
    :param Params: pipe parameters
    :return: pre=trained resNet model
    '''

    # using the built-in option of avg pooling of the last cov layer
    model_1 = ResNet50V2(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')

    # Taking the output of the ResNet50 vector
    last_layer = model_1.output

    # adding the output layer using the sigmoid function to get probability
    predictions = Dense(1, activation='sigmoid')(last_layer)

    # Model to be trained
    model = Model(inputs=model_1.input, outputs=predictions)

    # Train only the layers which we have added at the end
    for layer in model_1.layers:
        layer.trainable = False

    optimizer = SGD(learning_rate=Params['Optimizer']['SGD']['learning_rate'],
                    momentum=Params['Optimizer']['SGD']['momentum'], nesterov=Params['Optimizer']['SGD']['nesterov'])

    # using SGD(stochastic gradient descent)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

    return model


def train(model, train_x, train_y, Params):
    '''
      train a given model
      :param train_set
      :param: Params: pipe parameters
      :return: trained model
    '''

    model.fit(train_x, train_y, batch_size=Params['Model']['batch_size'], epochs=Params['Model']['epochs'])

    return model


def convert_to_binary_classification(output_values, threshold, valueIndx):
    '''
     converts float values of prediction into binary result
     based on a given threshold
     :param output_values: float values of prediction
     :param threshold: thershold for decision
     :param valueIndx: determine what is the index of the right values
     :return: predition in binary result
   '''

    predictions = []

    for value in output_values:
        if value[valueIndx] > threshold:
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions


def test(model, TestDataRep_x, Params):
    '''
      perform prediction of the trained model
      :param Model: trained classifier
      :param TestDataRep_x: testing feature vectors
      :param: Params: pipe parameters
      :return: predition (0/1) + the model output values (float)
    '''

    output_values = model.predict(TestDataRep_x)
    predictions = convert_to_binary_classification(output_values, Params['Model']['threshold'], 0)

    return np.array(predictions), output_values


def evaluate(Results, TestData, TestDataRep_y, output_values):
    '''
         calculate error +confusion matrix and display worst classified images
         :param Results: predictions vector
         :param TestDataRep_y: real values vector
         :param output_values: probabilities vector
         :param displayWorst: boolean
         :return: dictionary with the error rate and the confusion matrix
   '''

    error_rate = round(1 - accuracy_score(TestDataRep_y, Results), 3)
    conf_matrix = confusion_matrix(TestDataRep_y, Results, labels=[1, 0])

    summary = {'error_rate': error_rate, 'conf_matrix': conf_matrix}


    return summary


def ReportResults(Summary, output_values, TestDataRep_y, displayPlots, start_time, Params):
    '''
    printing the summary of results and display plots (conf_matrix + precision-recall curve)
    :param   Summary: dictionary with train error,validation error,Confusion matrix
    :param   output_values: vector of predictions
    :param   TestDataRep_y: test set labels
    :param   displayPlots: boolean
    :param   start_time: starting time of program
    :return None
    '''

    print('\n')
    print('|---------------------------------------|')
    print('|-------------|  Results  |-------------|')
    print('|---------------------------------------|')

    print('\nTest Error: {}'.format(Summary['error_rate']))
    print('\nConfusion Matrix: \n{}'.format(Summary['conf_matrix']))

    printRunTime(start_time)

    if displayPlots:
        # plot the precision-recall curve
        precision, recall, _ = precision_recall_curve(TestDataRep_y, output_values, pos_label=1)
        # plt.style.use('dark_background')
        fig = plt.figure(figsize=(10, 8))

        plt.subplot(2, 2, 1)
        # plt.style.use('dark_background')
        plt.plot(recall, precision, color='C4')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve')

        # plot the confusion matrix
        ax = plt.subplot(2, 2, 2)
        # plt.style.use('dark_background')
        sns.heatmap(Summary['conf_matrix'], annot=True, ax=ax, cmap='Blues')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(['Flower', 'Not Flower'])
        ax.yaxis.set_ticklabels(['Flower', 'Not Flower'])

        fig.subplots_adjust(wspace=0.5)
        fig.suptitle('Results Report')
        plt.show()


def printRunTime(strat_time):
    '''
     print the total running time
     :param   strat_time: starting timestamp
     '''

    end = time.time()
    temp = end - strat_time
    minutes = temp // 60
    seconds = temp - 60 * minutes
    print('\nTotal Run Time: %d:%d minutes' % (minutes, seconds))






def main():
    np.random.seed(0)
    Params = GetDefaultParameters()

    data_path = 'C:\\Users\\yonat\\Desktop\\Class\\YEAR D\\Semester A\\VR\\Project\\Task_2\\FlowerData'
    test_images_indices = list(range(301, 473))

    Params['Data']['path'] = data_path
    Params['Data']['test_images_indices'] = test_images_indices

    start_time = time.time()
    DandL = GetData(Params, False)
    # DandL = pd.read_pickle('fold1_raw_data.pkl')

    TrainData, TestData = TrainTestSplit(Params, DandL)
    print('Train - Test Sizes:\ntrain: {} \ntest: {}'.format(TrainData.shape[0], TestData.shape[0]))

    TrainDataRep_x, TrainDataRep_y = prepare(Params, TrainData)
    TestDataRep_x, TestDataRep_y = prepare(Params, TestData)

    # data augmentation
    # TrainDataRep_x, TrainDataRep_y = DataAugment(TrainDataRep_x, TrainDataRep_y, Params['Prepare']['dataAugment'])

    model = build_baseline_model(Params)

    trained_model = train(model, TrainDataRep_x, TrainDataRep_y, Params)

    predictions, output_values = test(trained_model, TestDataRep_x, Params)

    Summary = evaluate(predictions, TestData, TestDataRep_y, output_values)

    ReportResults(Summary, output_values, TestDataRep_y, True, start_time, Params)


if __name__ == '__main__':
    main()


