import numpy as np
import pandas as pd
from scipy.misc import imread, imresize

from keras.models import model_from_json
import json
from keras import optimizers


def read_imgs(img_paths):
    """
    read images from file path.
    """        
    imgs = np.empty([len(img_paths), 160, 320, 3])

    for i, path in enumerate(img_paths):
        imgs[i] = imread(path)

    return imgs

def resize(X):
    import tensorflow
    return tensorflow.image.resize_images(X, (40, 160))

def save_model(model):
    with open('model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
    model.save_weights('model.h5')

    print ('Model saved!')
    
    return True

def get_old_model(model_name):
    
    model_path = model_name
    with open(model_path, 'r') as jfile:
        model = model_from_json(json.loads(jfile.read()))

    weights_path = model_path[0:-5]+'.h5'
    #Compile with a slow learning rate
    model.compile(optimizer=optimizers.SGD(lr=1e-5, momentum=0.9), loss='mse')
    model.load_weights(weights_path)
    
    print (model_name+' loaded!')
    
    return model

def rewrite_data(df):
	#Re-write new paths as old paths.
	str_rpl = '/home/tyler/programming/behavioral-cloning/'
	df['center_imgpath'] = df['center_imgpath'].str.replace(str_rpl, '')
	df['right_imgpath'] = df['right_imgpath'].str.replace(str_rpl, '')
	df['left_imgpath'] = df['left_imgpath'].str.replace(str_rpl, '')
	df.to_csv('driving_log.csv', index=False)
	print("Data saved.")


def main():

	# 1. Read Data from Driving Log
	driving_log = pd.read_csv('driving_log.csv', index_col=False)
	driving_log.columns = ['center_imgpath', 'left_imgpath', 'right_imgpath', 'angle', 'throttle', 'break', 'speed']

	# 2. Prepare Data.
	newX_train_path, y_train = [], []

	for index, row in driving_log.iterrows():    
	    #Set angles for center, left and right cam   
	    C = row['angle']
	    L = C + 0.07
	    R = C - 0.07
	    
	    #only append the new stuff
	    try:
	        newX_train_path.append(row['center_imgpath'].split('cloning/',1)[1])
	        newX_train_path.append(row['right_imgpath'].split('cloning/',1)[1])
	        newX_train_path.append(row['left_imgpath'].split('cloning/',1)[1])
	        y_train.append(C)
	        y_train.append(R)
	        y_train.append(L)
	    except IndexError:
	        pass
	    
	    row['center_imgpath'] = (row['center_imgpath'].split('cloning/',1)[-1])
	    row['right_imgpath'] = (row['right_imgpath'].split('cloning/',1)[-1])
	    row['left_imgpath'] = (row['left_imgpath'].split('cloning/',1)[-1])
	    
	#convert to np array
	newX_train_path, y_train = np.array(newX_train_path), np.array(y_train)

	if len(newX_train_path) != 0 :
	    X_train = read_imgs(newX_train_path)
	    model = get_old_model('model.json')   
	    #Re-train old model
	    model.fit(X_train,
	              y_train,
	              batch_size=32,
	              nb_epoch=5,
	              validation_split=0.0)
	    save_model(model)
	    rewrite_data(driving_log)
	else:
	    print("No new data!")	

if __name__ == '__main__':
	main()
    
