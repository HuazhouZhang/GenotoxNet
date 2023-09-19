import argparse
import random,os,sys
import numpy as np
import csv
from scipy import stats
import time
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import pandas as pd
import keras
import keras.backend as K
from keras.models import Model, Sequential
from keras.models import load_model
from keras.layers import Input,InputLayer,Multiply,ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense,Activation,Dropout,Flatten,Concatenate
from keras.layers import BatchNormalization
from keras.layers import Lambda
from keras import optimizers,utils
from keras.constraints import max_norm
from keras import regularizers
from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping,History,CSVLogger,ReduceLROnPlateau
from keras.utils import plot_model
from keras.optimizers import Adam, SGD
from keras.models import model_from_json
import tensorflow.compat.v1 as tf
from sklearn.metrics import average_precision_score
from scipy.stats import pearsonr
from Genotoxicity_model import KerasMultiSourceGCNModel
import hickle as hkl
import scipy.sparse as sp
import argparse
from sklearn.model_selection import KFold
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import matplotlib.pyplot as plt

####################################Settings#################################
parser = argparse.ArgumentParser(description='Chemical_Genotoxicity_pre')
parser.add_argument('-gpu_id', dest='gpu_id', type=str, default='0', help='GPU devices')
parser.add_argument('-use_gexp', dest='use_gexp', type=bool, default=True, help='use gene expression or not')
parser.add_argument('-use_assay', dest='use_assay', type=bool, default=True, help='use assay or not')
parser.add_argument('-israndom', dest='israndom', type=bool, default=False, help='randomlize X and A')
#hyparameters for GCN
parser.add_argument('-unit_list', dest='unit_list', nargs='+', type=int, default=[256,256,256],help='unit list for GCN')
parser.add_argument('-use_bn', dest='use_bn', type=bool, default=True, help='use batchnormalization for GCN')
parser.add_argument('-use_relu', dest='use_relu', type=bool, default=True, help='use relu for GCN')
parser.add_argument('-use_GMP', dest='use_GMP', type=bool, default=True, help='use GlobalMaxPooling for GCN')
args = parser.parse_args()
random.seed(0)
tf.set_random_seed(1)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(2)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_gexp = args.use_gexp
use_assay = args.use_assay
israndom=args.israndom
model_suffix = ('with_gexp' if use_gexp else 'without_gexp')
model_suffix1 = ('with_assay' if use_assay else 'without_assay')
GCN_deploy = '_'.join(map(str,args.unit_list)) + '_'+('bn' if args.use_bn else 'no_bn')+'_'+('relu' if args.use_relu else 'tanh')+'_'+('GMP' if args.use_GMP else 'GAP')
model_suffix = model_suffix + '_' + model_suffix1 + '_' +GCN_deploy

####################################Constants Settings###########################
Drug_feature_file = r'GenotoxNet_data\drug_graph_feat'
Gene_expression_file = r'GenotoxNet_data\TSdata.csv'
ToxCast_Assay_file = r'GenotoxNet_data\Assaydata.csv'
Max_atoms = 100

def DataGenerate(Drug_feature_file,Gene_expression_file,ToxCast_Assay_file):
    # load drug features
    drug_pubchem_id_set = []
    all_drug_feature = {}
    for each in os.listdir(Drug_feature_file):
        drug_pubchem_id_set.append(each.split('.')[0])
        feat_mat,adj_list,degree_list = hkl.load('%s/%s'%(Drug_feature_file,each))
        all_drug_feature[each.split('.')[0]] = [feat_mat,adj_list,degree_list]
    assert len(drug_pubchem_id_set)==len(all_drug_feature.values())
    #load gene expression faetures
    all_gexpr_feature = pd.read_csv(Gene_expression_file,sep=',',header=0,index_col=[0]).T
    all_assay_feature = pd.read_csv(ToxCast_Assay_file,sep=',',header=0,index_col=[0]).T
    return all_drug_feature,all_gexpr_feature,all_assay_feature

def MetadataGenerate(all_drug_feature,all_gexpr_feature,all_assay_feature):
    label = pd.read_csv(r'Dataset\Genotoxicity_Training_set.csv',index_col = None, header = 0)
    drugnames = label['pert_id'].tolist()
    label['Genotoxicity_label'] = (label['Genotoxicity'] == '+').astype(int)
    data_idx = list(zip(label['pert_id'],label['Genotoxicity_label']))
    nb_drugs = len(set([item[0] for item in data_idx]))
    drug_feature = {key: all_drug_feature[key] for key in drugnames} 
    gexpr_feature = all_gexpr_feature.loc[drugnames,:]
    assay_feature = all_assay_feature.loc[drugnames,:]
    print('%d instances across and %d chemicals were generated.'%(len(data_idx),nb_drugs))
    return drug_feature,gexpr_feature,assay_feature,data_idx

def TestGenerate(all_drug_feature,all_gexpr_feature,all_assay_feature):
    label = pd.read_csv(r'Dataset\Genotoxicity_Test_set.csv',index_col = None, header = 0)
    drugnames = label['pert_id'].tolist()
    label['Genotoxicity_label'] = (label['Genotoxicity'] == '+').astype(int)
    data_idx = list(zip(label['pert_id'],label['Genotoxicity_label']))
    nb_drugs = len(set([item[0] for item in data_idx]))
    drug_feature = {key: all_drug_feature[key] for key in drugnames} 
    gexpr_feature = all_gexpr_feature.loc[drugnames,:]
    assay_feature = all_assay_feature.loc[drugnames,:]
    return drug_feature,gexpr_feature,assay_feature,data_idx

def NormalizeAdj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm

def random_adjacency_matrix(n):
    matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]
    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 0
    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]
    return matrix

def CalculateGraphFeat(feat_mat,adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    feat = np.zeros((Max_atoms,feat_mat.shape[-1]),dtype='float32')
    adj_mat = np.zeros((Max_atoms,Max_atoms),dtype='float32')
    if israndom:
        feat = np.random.rand(Max_atoms,feat_mat.shape[-1])
        adj_mat[feat_mat.shape[0]:,feat_mat.shape[0]:] = random_adjacency_matrix(Max_atoms-feat_mat.shape[0])        
    feat[:feat_mat.shape[0],:] = feat_mat
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i,int(each)] = 1
    assert np.allclose(adj_mat,adj_mat.T)
    adj_ = adj_mat[:len(adj_list),:len(adj_list)]
    adj_2 = adj_mat[len(adj_list):,len(adj_list):]
    norm_adj_ = NormalizeAdj(adj_)
    norm_adj_2 = NormalizeAdj(adj_2)
    adj_mat[:len(adj_list),:len(adj_list)] = norm_adj_
    adj_mat[len(adj_list):,len(adj_list):] = norm_adj_2    
    return [feat,adj_mat]

def FeatureExtract(data_idx,drug_feature,gexpr_feature,assay_feature):
    nb_instance = len(data_idx)
    nb_gexpr_features = gexpr_feature.shape[1]
    nb_assay_features = assay_feature.shape[1]
    drug_data = [[] for item in range(nb_instance)]
    gexpr_data = np.zeros((nb_instance,nb_gexpr_features),dtype='float32') 
    assay_data = np.zeros((nb_instance,nb_assay_features),dtype='int16')
    target = np.zeros(nb_instance,dtype='int16')
    for idx in range(nb_instance):
        drugname,binary_IC50 = data_idx[idx]
        #modify
        feat_mat,adj_list,_ = drug_feature[str(drugname)]
        #fill drug data,padding to the same size with zeros
        drug_data[idx] = CalculateGraphFeat(feat_mat,adj_list)
        #randomlize X A
        gexpr_data[idx,:] = gexpr_feature.loc[drugname,:]
        assay_data[idx,:] = assay_feature.loc[drugname,:]
        target[idx] = binary_IC50
    # return drug_data,gexpr_data,assay_data,toxicity
    return drug_data,gexpr_data,assay_data,target

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    recal = recall(y_true, y_pred)
    return 2.0*prec*recal/(prec+recal+K.epsilon())

def average_precision(y_true, y_pred):
    return tf.py_function(average_precision_score, (y_true, y_pred), tf.double) 

class MyCallback(Callback):
    def __init__(self,validation_data,patience):
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.best_weight = None
        self.patience = patience
    def on_train_begin(self,logs={}):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf
        self.losses = {'batch':[]}
        self.val_loss = {'epoch':[]}
        self.aucl = {'epoch':[]}
        return
    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        return
    def on_train_end(self, logs={}):
        self.model.set_weights(self.best_weight)
        self.model.save('GenotoxNet_data/bestmodel/MyGenotoxNet_classify_%s.h5'%model_suffix)
        if self.stopped_epoch > 0 :
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
        return
    def on_epoch_begin(self, epoch, logs={}):
        return
    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        precision,recall,_, = metrics.precision_recall_curve(self.y_val,y_pred_val)
        pr_val = -np.trapz(precision,recall)
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.aucl['epoch'].append(roc_val)
        print('roc-val: %.4f, pr-val:%.4f' % (roc_val,pr_val))
        if roc_val > self.best:
            self.best = roc_val
            self.wait = 0
            self.best_weight = self.model.get_weights()
            self.model.save('GenotoxNet_data/bestmodel/MyGenotoxNet_highestAUCROC_%s.h5'%model_suffix)
        else:
            self.wait+=1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
        return
    def savedata(self,lr,batchsize,wd,dd):
        iters = range(len(self.val_loss['epoch']))
        eponb = float(len(self.losses['batch']))/float(len(self.val_loss['epoch']))
        dflist = []
        for ii in iters:
            ystart = int(ii*eponb)
            yend = int((ii+1)*eponb)
            yloss = float(np.sum(self.losses['batch'][ystart:yend]))/float(eponb)
            testloss = self.val_loss['epoch'][ii]
            aucroc = self.aucl['epoch'][ii]
            dflist.append([ii+1,aucroc,yloss,testloss])
        df = pd.DataFrame(dflist)
        df.columns = ['epoch','auc','train_loss','validation_loss']
        df.to_csv('GenotoxNet_data/bestmodel/lr%s_batch%s_dropout%s_%sl2_loss.csv'%(lr,batchsize,dd,wd),index=False,header=True)
        return

def ModelTraining(model,lr,batchsize,wd,dd,X_drug_data_train,X_gexpr_data_train,X_assay_data_train,Y_train,validation_data,nb_epoch=300):
    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer = optimizer,loss='binary_crossentropy',metrics=['accuracy',precision,recall,f1_score,average_precision])
    X_drug_feat_data_train = [item[0] for item in X_drug_data_train]
    X_drug_adj_data_train = [item[1] for item in X_drug_data_train]
    X_drug_feat_data_train = np.array(X_drug_feat_data_train)#nb_instance * Max_stom * feat_dim
    X_drug_adj_data_train = np.array(X_drug_adj_data_train)#nb_instance * Max_stom * Max_stom
    train_validation_data = [[X_drug_feat_data_train,X_drug_adj_data_train,X_gexpr_data_train,X_assay_data_train],Y_train]
    history = MyCallback(validation_data=validation_data,patience=20)
    callbacks = [ModelCheckpoint(filepath='GenotoxNet_data/bestmodel/Genotoxicity_weights_{epoch:04d}.h5',verbose=1),history]
    model.fit(x=[X_drug_feat_data_train,X_drug_adj_data_train,X_gexpr_data_train,X_assay_data_train],y=Y_train,batch_size=batchsize,epochs=nb_epoch,validation_data=(validation_data[0],validation_data[1]),callbacks=callbacks)
    history.savedata(lr,batchsize,wd,dd)
    return model

def ModelEvaluate(model,X_drug_data_test,X_gexpr_data_test,X_assay_data_test,Y_test,cancer_type_test_list):
    X_drug_feat_data_test = [item[0] for item in X_drug_data_test]
    X_drug_adj_data_test = [item[1] for item in X_drug_data_test]
    X_drug_feat_data_test = np.array(X_drug_feat_data_test)#nb_instance * Max_stom * feat_dim
    X_drug_adj_data_test = np.array(X_drug_adj_data_test)#nb_instance * Max_stom * Max_stom    
    Y_pred = model.predict([X_drug_feat_data_test,X_drug_adj_data_test,X_gexpr_data_test,X_assay_data_test])
    auROC_all = metrics.roc_auc_score(Y_test, Y_pred)
    fpr,tpr,_,= metrics.roc_curve(Y_test,Y_pred)
    precision,recall,_, = metrics.precision_recall_curve(Y_test,Y_pred)
    auPR_all = -np.trapz(precision,recall)
    print("The overall AUC and auPR is %.4f and %.4f."%(auROC_all,auPR_all))
    return auROC_all,auPR_all,Y_pred

def main():
    dflist = []
    lr = 0.0001
    batchsize = 16
    dd = 0.1
    wd = 0.0001
    print(lr,batchsize,dd,wd)
    sed = 225
    random.seed(0)
    tf.set_random_seed(1)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    np.random.seed(2)
    datatype = 'all'
    all_drug_feature,all_gexpr_feature,all_assay_feature = DataGenerate(Drug_feature_file,Gene_expression_file,ToxCast_Assay_file)
    #Extract features for training, development and test 
    drug_feature,gexpr_feature,assay_feature,data_idx = MetadataGenerate(all_drug_feature,all_gexpr_feature,all_assay_feature)
    X_drug_data_train,X_gexpr_data_train,X_assay_data_train,Y_train = FeatureExtract(data_idx,drug_feature,gexpr_feature,assay_feature)
    drug_test_feature,gexpr_test_feature,assay_test_feature,data_test_idx = TestGenerate(all_drug_feature,all_gexpr_feature,all_assay_feature)
    data_idx_1 = [a for a in data_test_idx if a[1]==1]
    data_idx_0 = [a for a in data_test_idx if a[1]==0]
    random.seed(sed)
    data_idx_validation = random.sample(data_idx_1,int(0.5*len(data_idx_1)))+random.sample(data_idx_0,int(0.5*len(data_idx_0)))
    data_idx_extra = [a for a in data_test_idx if a not in data_idx_validation]
    print(data_idx_extra)
    random.seed(0)
    X_drug_data_validation,X_gexpr_data_validation,X_assay_data_validation,Y_validation = FeatureExtract(data_idx_validation,drug_test_feature,gexpr_test_feature,assay_test_feature)
    X_drug_feat_data_validation = [item[0] for item in X_drug_data_validation]
    X_drug_adj_data_validation = [item[1] for item in X_drug_data_validation]
    X_drug_feat_data_validation = np.array(X_drug_feat_data_validation)#nb_instance * Max_stom * feat_dim
    X_drug_adj_data_validation = np.array(X_drug_adj_data_validation)#nb_instance * Max_stom * Max_stom  
    validation_data = [[X_drug_feat_data_validation,X_drug_adj_data_validation,X_gexpr_data_validation,X_assay_data_validation],Y_validation]
    model = KerasMultiSourceGCNModel(use_gexp,use_assay,regr=False).createMaster(
    X_drug_data_train[0][0].shape[-1],X_gexpr_data_train.shape[-1],X_assay_data_train.shape[-1],
    args.unit_list,wd,dd,args.use_relu,args.use_bn,args.use_GMP)
    print('Begin training...')
    model = ModelTraining(model,lr,batchsize,wd,dd,X_drug_data_train,X_gexpr_data_train,X_assay_data_train,Y_train,validation_data,nb_epoch=300)    
    X_drug_data_extra,X_gexpr_data_extra,X_assay_data_extra,Y_extra = FeatureExtract(data_idx_extra,drug_test_feature,gexpr_test_feature,assay_test_feature)
    X_drug_feat_data_extra = [item[0] for item in X_drug_data_extra]
    X_drug_adj_data_extra = [item[1] for item in X_drug_data_extra]
    X_drug_feat_data_extra = np.array(X_drug_feat_data_extra)#nb_instance * Max_stom * feat_dim
    X_drug_adj_data_extra = np.array(X_drug_adj_data_extra)#nb_instance * Max_stom * Max_stom   
    auROC_train,auPR_train,Y_pred_train = ModelEvaluate(model,X_drug_data_train,X_gexpr_data_train,X_assay_data_train,Y_train,r'GenotoxNet_data\GenotoxNet_train_%s.log'%(model_suffix))
    auROC_validation,auPR_validation,Y_pred_val = ModelEvaluate(model,X_drug_data_validation,X_gexpr_data_validation,X_assay_data_validation,Y_validation,r'GenotoxNet_data\GenotoxNet_validation_%s.log'%(model_suffix))
    auROC_extra,auPR_extra,Y_pred_extra = ModelEvaluate(model,X_drug_data_extra,X_gexpr_data_extra,X_assay_data_extra,Y_extra,r'GenotoxNet_data\GenotoxNet_extra_%s.log'%(model_suffix))
    testres = pd.DataFrame([Y_test,Y_pred_extra])
    testres.to_csv('GenotoxNet_data/predict_result/lr%s_batch%s_dropout%s_%sl2_extra_res.csv'%(lr,batchsize,dd,wd),index=False,header=False)
    print(auROC_train,auPR_train)
    print(auROC_validation,auPR_validation)
    print(auROC_extra,auPR_extra)
    dflist.append([auROC_train,auROC_validation,auROC_extra])
    df = pd.DataFrame(dflist)
    df.columns = ['auROC_train','auROC_validation','auROC_extra']
    df.to_csv('GenotoxNet_data/train_validation_test_result.csv',index=False)

if __name__=='__main__':
    main()