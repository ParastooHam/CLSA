from layer.output_layer import *
from layer.Decoder import *
from layer.Encoder import *
from utils import *
import tensorflow as tf
import copy
import pandas as pd
import numpy as np
import itertools
import datetime
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
from datetime import timedelta
from collections import  Counter
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import OneHotEncoder
import random
import matplotlib.pyplot as plt
previous_visit = 0
predicted_visit = 1
batch_size = 256
epochs = 1
MAX_CINDEX = 0
hidden_size1_final = 0
hidden_size2_final = 0
hidden_size3_final = 0
hidden_size4_final = 0
a1_final = 0
a2_final = 0
a3_final = 0
learning_rate_final = 0
l2_regularization_final = 0
MASK_RATE = 0
SHUFFLE_RATE = 1
precision = np.zeros((2, 5))
recall = np.zeros((2, 5))
score = np.zeros((2, 5))

Report = []
def train_model(train_set, test_set, feature_dims, hidden_size, num_category,
                    num_event, learning_rate, l2_regularization, MASK_RATE, SHUFFLE_RATE, ith_fold):
    feature = train_set.x
    visit_len = feature.shape[1]
    encoder = Encoder(hidden_size=512, model_type='LSTM')
    fc_net = []
    for i in range(num_event):
        fc_net.append(FC_SAP(hidden_size=hidden_size, num_category=num_category))
    FC = FC_SAP(hidden_size=hidden_size, num_category=num_category)
    decoder = Decoder(hidden_size=512, feature_dims=feature_dims, model_type='TimeLSTM2')
    mlp = MLP2(hidden_size=num_category*num_event)
    logged = set()
    result = []
    result_index = []
    result2 = []
    result2_index = []
    shuffle_index = [19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4, 3, 2, 1, 0]
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.99, beta_2=0.999, epsilon=None, decay=0.000001, amsgrad=False, clipnorm=1.)
    while train_set.epoch_completed < epochs:
        input_x_train_, input_t_train_, input_y_train_, input_day_train_, input_mask1_train_, input_mask2_train_, input_mask3_train_ = train_set.next_batch(batch_size)
        visit_len = input_x_train_.shape[1]
        with tf.GradientTape() as tape:
            mask_index = (visit_len - 1)
            if MASK_RATE == 0:
                mask_input_x_train = copy.deepcopy(input_x_train_)
            else:
                mask_input_x_train = copy.deepcopy(input_x_train_)
                random_select = np.random.random()
                random_select_list = np.array(range(mask_input_x_train.shape[0]))
                np.random.shuffle(random_select_list)
                random_select_list_sort = pd.Series(random_select_list).sort_values()
                random_select_list_sort_index = random_select_list_sort.index[
                                                :int(mask_input_x_train.shape[0] * MASK_RATE)]
                mask_input_x_train[random_select_list_sort_index, mask_index, :] = 0
            trajectory_encode_last_h, trajectory_encode_h_list = encoder(mask_input_x_train, batch=batch_size)
            predicted_trajectory_x_train, predicted_trajectory_decode_h = decoder(
                (trajectory_encode_last_h, input_day_train_),
                predicted_visit=visit_len,
                batch=batch_size)
            gen_mse_loss = tf.reduce_mean(
                tf.keras.losses.mse(input_x_train_[:, mask_index, :], predicted_trajectory_x_train[:, mask_index, :]))
            mask_input_x_train_add = copy.deepcopy(mask_input_x_train)
            mask_input_x_train_add[:, mask_index, :] = predicted_trajectory_x_train[:, mask_index, :]
            mask_input_x_train_trajectory_generation_decode_h, mask_input_x_train_trajectory_generation_h_list = encoder(
                mask_input_x_train_add,
                batch=batch_size)
            real_decode_h, real_trajectory_encode_h_list = encoder(input_x_train_, batch=batch_size)
            clf_loss = 0
            neg_likelihood_loss = 0
            predicted_output = []
            for i in range(num_event):
                label = input_y_train_[:,-1].reshape((-1, 1)).astype('float32')
                ett = input_t_train_[:,-1].reshape((-1, 1)).astype('float32')
                predicted_output_ = fc_net[i](real_decode_h)
                predicted_output.append(predicted_output_)
            out = tf.stack(predicted_output, axis=1)
            out = tf.reshape(out, [-1, num_event * hidden_size])
            out = mlp(out)
            out = tf.reshape(out, [-1, num_event, num_category])
            for i in range(num_event):
                predicted_output_ = out[:,i,:]
                I_2 = np.cast['float32'](np.equal(label, i + 1))
                denom = 1 - tf.reduce_sum(input_mask1_train_[:,i,:] * predicted_output_, axis=1)
                denom = tf.clip_by_value(denom, tf.cast( 1e-08, dtype=tf.float32),
                                         tf.cast(1. - 1e-08, dtype=tf.float32))
                tmp2 = tf.reduce_sum(input_mask2_train_[:,i,:] * predicted_output_, axis=1),
                tmp2 =  log(div(tmp2,denom))
                neg_likelihood_loss += - tf.reduce_mean(tmp2)
            if SHUFFLE_RATE == 0:
                shuffled_input_x_train = input_x_train_
            else:
                random_select_list2 = np.array(range(mask_input_x_train.shape[0]))
                np.random.shuffle(shuffle_index)
                random_select_list_sort2 = pd.Series(random_select_list2).sort_values()
                random_select_list_sort_index2 = random_select_list_sort2.index[
                                                 :int(mask_input_x_train.shape[0] * SHUFFLE_RATE)]
                shuffled_input_x_train_mask1 = np.ones_like(input_x_train_)
                shuffled_input_x_train_mask1[random_select_list_sort_index2, :, :] = 0
                shuffled_input_x_train_mask0 = np.zeros_like(input_x_train_)
                shuffled_input_x_train_mask0[random_select_list_sort_index2, :, :] = 1
                shuffled_input_x_train = (input_x_train_ * shuffled_input_x_train_mask0)[:, shuffle_index,
                                         :] + input_x_train_ * shuffled_input_x_train_mask1
            shuffled_generated_decode_h, shuffled_generated_decode_h_list = encoder((shuffled_input_x_train),
                                                                                    batch=batch_size)
            contrast_loss_matrix = tf.matmul(shuffled_generated_decode_h, tf.transpose(real_decode_h))
            contrast_loss_numerator = tf.linalg.diag_part(contrast_loss_matrix)
            contrast_loss_denominator = tf.reduce_sum(tf.math.exp(contrast_loss_matrix),
                                                      axis=1)
            contrast_loss = -tf.reduce_mean(contrast_loss_numerator - tf.math.log(contrast_loss_denominator))
            contrast_loss_trajectory_generation = tf.matmul(mask_input_x_train_trajectory_generation_decode_h,
                                                            tf.transpose(real_decode_h))
            contrast_loss_trajectory_generation_numerator = tf.linalg.diag_part(contrast_loss_trajectory_generation)
            contrast_loss_trajectory_generation_denominator = tf.reduce_sum(
                tf.math.exp(contrast_loss_trajectory_generation),
                axis=1)
            contrast_loss2 = -tf.reduce_mean(
                contrast_loss_trajectory_generation_numerator - tf.math.log(
                    contrast_loss_trajectory_generation_denominator))
            contrast_loss_risk = 0
            for i in range(num_event):
                h_e = tf.gather(real_decode_h, tf.where(label==i+1)[:,0])
                h_0 = tf.gather(real_decode_h, tf.where(label!=i+1)[:,0])
                contrast_loss_risk_numerator = tf.matmul(h_e, tf.transpose(h_e))
                contrast_loss_risk_denominator = tf.math.exp(tf.matmul(h_e, tf.transpose(h_0)))
                contrast_loss_risk += -tf.reduce_sum(contrast_loss_risk_numerator-tf.math.log(tf.reduce_sum(contrast_loss_risk_denominator)))
            whole_loss =  gen_mse_loss * 0.2 +neg_likelihood_loss * 0.5 + contrast_loss * 0.15 + contrast_loss2 * 0.15 #0.5 *contrast_loss_risk +
            fc_net_variables = []
            for i in range(num_event):
                fc_net_variables.extend([var for var in fc_net[i].trainable_variables])
            mlp_variables = [var for var in mlp.trainable_variables]
            for weight in mlp.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            encoder_variables = [var for var in encoder.trainable_variables]
            for weight in encoder.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            decoder_variables = [var for var in decoder.trainable_variables]
            for weight in decoder.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            variables =mlp_variables + encoder_variables + fc_net_variables + decoder_variables
            gradient = tape.gradient(whole_loss, variables)
            optimizer.apply_gradients(zip(gradient, variables))
            if train_set.epoch_completed not in logged:
                logged.add(train_set.epoch_completed)
                input_x_test = test_set.x
                input_y_test = test_set.y
                input_t_test = test_set.t
                input_day_test = test_set.day
                batch_test = input_x_test.shape[0]
                context_state_test, real_trajectory_encode_h_list_test = encoder(input_x_test, batch=batch_test)
                c_index_output = []
                accuracy = []
                label_test = input_y_test[:,-1].reshape((-1, 1)).astype('float32')
                ett_test = input_t_test[:,-1].reshape((-1, 1)).astype('float32')
                day_test = input_day_test[:,-1].reshape((-1, 1)).astype('float32')
                last_meas_ = ett_test
                predicted_o_list = []
                for i in range(num_event):
                    predicted_o_list.append(fc_net[i](context_state_test))
                out = np.stack(predicted_o_list, axis=1)
                out = np.reshape(out, [-1, num_event * hidden_size])
                out = mlp(out)
                out = np.reshape(out, [-1, num_event, num_category])
                for i in range(num_event):
                    predicted_output_test = out[:,i,:]
                    predicted_risk_test = f_get_risk_predictions(predicted_output_test, last_meas_, 4)
                    predicted_output.append(predicted_risk_test)
                    I_2 = np.cast['float32'](np.equal(label_test, i + 1))
                    y_pred_label, auc_test, precision_test, recall_test, f_score_test, accuracy_test = calculate_score(I_2,
                                                                                                     predicted_risk_test,print_flag=True )
                    report = classification_report(I_2, y_pred_label, output_dict = True)
                    precision[0,ith_fold] = report['0.0']['precision']
                    precision[1,ith_fold] = report['1.0']['precision']
                    recall[0,ith_fold] = report['0.0']['recall']
                    recall[1,ith_fold] = report['1.0']['recall']
                    score[0,ith_fold] = report['0.0']['f1-score']
                    score[1,ith_fold] = report['1.0']['f1-score']
                    Report.append(report)
                    accuracy.append(accuracy_test)
                pre0 = report['0.0']['precision']
                pre1 = report['1.0']['precision']
                rec0 = report['0.0']['recall']
                rec1 = report['1.0']['recall']
                sco0 = report['0.0']['f1-score']
                sco1 = report['1.0']['f1-score']
                accuracy.append(np.mean(accuracy))
                result2.append(accuracy)
                result2_index.append(np.sum(accuracy))
                print('----epoch:{}, whole_loss:{}, contrast_loss:{},contrast_loss2:{},clf_loss:{},gen_loss:{}, acc:{}'.format(train_set.epoch_completed, whole_loss, contrast_loss, contrast_loss2, neg_likelihood_loss,
                                                                                                                                           gen_mse_loss, accuracy[0], pre0, pre1))
        tf.compat.v1.reset_default_graph()
    result2 = np.array(result2)
    result2_index = np.array(result2_index)
    max_i2 = np.where(result2_index == result2_index.max())
    res = result2[max_i2[0], 0]
    if len(res) >=2:
        ans = res[0]
    else:
        ans = result2[max_i2[0], 0]
    return ans , pre0, pre1 , rec0, rec1, sco0, sco1,  encoder, FC, decoder, mlp

def f_get_risk_predictions(pred, last_meas_, pre_time):
    _, num_Category = np.shape(pred)
    pred_s = np.zeros(np.shape(pred))
    pred_a = np.zeros(np.shape(pred))
    for i in range(pred_s.shape[0]):
        l = int(last_meas_[i][0])
        pred_s[i, l:(l+pre_time+1)] = pred[i, l:(l+pre_time+1)]
        pred_a[i,  l:] = pred[i,  l:]
    risk = np.sum(pred_s, axis=1)
    if np.sum(pred_a, axis=1) != np.zeros(np.shape(pred)):
        risk = risk / np.sum(pred_a, axis=1)
    return risk


def f_get_risk_predictions2(o_list, pred, last_meas_, pre_time):
    _, num_Category = np.shape(pred)
    pred_s = np.zeros(np.shape(pred))
    pred_a = np.zeros(np.shape(pred))
    for i in range(pred_s.shape[0]):
        l = int(last_meas_[i][0])
        pred_s[i, l:(l+pre_time+1)] = pred[i, l:(l+pre_time+1)]
        pred_a[i,  l:] = pred[i,  l:]
    risk = np.sum(pred_s, axis=1)
    risk = risk / (np.sum(pred_a, axis=1))
    return risk


def f_get_fc_mask1(meas_time, num_Event, num_Category):
    mask = np.zeros([np.shape(meas_time)[0], num_Event, num_Category])
    for i in range(np.shape(meas_time)[0]):
        mask[i, :, :int(meas_time[i]+1)] = 1
    return mask


def f_get_fc_mask2(time, label, num_Event, num_Category):
    mask = np.zeros([np.shape(time)[0], num_Event, num_Category])
    for i in range(np.shape(time)[0]):
        if label[i] != 0:
            mask[i,int(label[i]-1),int(time[i])] = 1
        else:
            mask[i,:,int(time[i]+1):] =  1
    return mask


def f_get_fc_mask3(time, meas_time, num_Category):
    mask = np.zeros([np.shape(time)[0], num_Category])
    if np.shape(meas_time):
        for i in range(np.shape(time)[0]):
            t1 = int(meas_time[i])
            t2 = int(time[i])
            mask[i,(t1+1):(t2+1)] = 1
    else:
        for i in range(np.shape(time)[0]):
            t = int(time[i])
            mask[i,:(t+1)] = 1
    return mask

def mimic_data_spilt(features, labels, event_num):
    split_ = []
    for i in range(event_num):
        features_i = features[i].reshape(features[i].shape[0], -1)
        labels_i = labels[i].reshape(labels[i].shape[0], -1)
        split_i = StratifiedShuffleSplit(n_splits =5,test_size= 0.2, train_size=0.8, random_state=1).split(features_i, labels_i)
        split_.append(split_i)
    return split_

def mimic_data_divided(data, visit_len, feature_dims):
    labels = data[:, -1].astype('float32').reshape(-1, visit_len,1)
    values, counts = np.unique(data[:, -1], return_counts=True)
    features = data[:, 0:28].astype('float32').reshape(-1, visit_len, feature_dims)
    days = data[:, -3].astype('float32').reshape(-1, visit_len) #follow ups
    ett = data[:, -2].astype('float32').reshape(-1, visit_len,1) #last measurement
    features_index = features.reshape(features.shape[0], -1)
    label_index = labels.reshape(labels.shape[0], -1)
    return features, labels, days, ett


def mimic_data_concat(split,features,labels, days, ett, mask1, mask2, mask3, event_num):
    train_index_0, test_index_0 = next(split[0])
    train_features = features[0][train_index_0]
    train_labels = labels[0][train_index_0]
    train_days = days[0][train_index_0]
    train_ett = ett[0][train_index_0]
    train_mask1 = mask1[0][train_index_0]
    train_mask2 = mask2[0][train_index_0]
    train_mask3 = mask3[0][train_index_0]
    test_features = features[0][test_index_0]
    test_labels = labels[0][test_index_0]
    test_days = days[0][test_index_0]
    test_ett = ett[0][test_index_0]
    test_mask1 = mask1[0][test_index_0]
    test_mask2 = mask2[0][test_index_0]
    test_mask3 = mask3[0][test_index_0]
    for i in range(event_num):
        train_index, test_index = next(split[i+1])
        train_features = np.concatenate([train_features, features[i+1][train_index]], axis=0)
        train_labels = np.concatenate([train_labels, labels[i + 1][train_index]], axis=0)
        train_days = np.concatenate([train_days, days[i + 1][train_index]], axis=0)
        train_ett = np.concatenate([train_ett, ett[i + 1][train_index]], axis=0)
        train_mask1 = np.concatenate([train_mask1, mask1[i + 1][train_index]], axis=0)
        train_mask2 = np.concatenate([train_mask2, mask2[i + 1][train_index]], axis=0)
        train_mask3 = np.concatenate([train_mask3, mask3[i + 1][train_index]], axis=0)
        test_features = np.concatenate([test_features, features[i + 1][test_index]], axis=0)
        test_labels = np.concatenate([test_labels, labels[i + 1][test_index]], axis=0)
        test_days = np.concatenate([test_days, days[i + 1][test_index]], axis=0)
        test_ett = np.concatenate([test_ett, ett[i + 1][test_index]], axis=0)
        test_mask1 = np.concatenate([test_mask1, mask1[i + 1][test_index]], axis=0)
        test_mask2 = np.concatenate([test_mask2, mask2[i + 1][test_index]], axis=0)
        test_mask3 = np.concatenate([test_mask3, mask3[i + 1][test_index]], axis=0)

    return DataSetWithMask2(train_features, train_ett, train_labels, train_days,train_mask1,train_mask2,train_mask3),DataSetWithMask2(test_features, test_ett, test_labels, test_days,test_mask1,test_mask2,test_mask3)
def mask_divide(label_,mask1,mask2,mask3):
    mask1 = [mask1[np.where(label_ == 0)],mask1[np.where(label_ == 1)]]
    mask2 = [mask2[np.where(label_ == 0)], mask2[np.where(label_ == 1)]]
    mask3 = [mask3[np.where(label_ == 0)], mask3[np.where(label_ == 1)]]
    return mask1,mask2,mask3
def experiment(MASK_RATE,SHUFFLE_RATE):
    train_repeat = 4
    ac_list =[]
    ac_List_total = np.zeros((train_repeat, 5))
    pre0_list = []
    pre1_list = []
    rec0_list = []
    rec1_list = []
    sco0_list = []
    sco1_list = []
    pre0_List_total = np.zeros((train_repeat, 5))
    pre1_List_total = np.zeros((train_repeat, 5))
    rec0_List_total = np.zeros((train_repeat, 5))
    rec1_List_total = np.zeros((train_repeat, 5))
    sco0_List_total = np.zeros((train_repeat, 5))
    sco1_List_total = np.zeros((train_repeat, 5))

    df_user = pd.read_csv("user_info.csv", delimiter=',', keep_default_na=False)
    df_rank = pd.read_csv('Rank.csv')
    df_rank_user = pd.merge(df_rank, df_user, on='user_id')
    df = df_rank_user
    N_Patients = np.max(df['FileID'])
    values, counts = np.unique(df['FileID'], return_counts=True)
    df = df.sort_values(by=['FileID', 'time'])
    enc = OneHotEncoder(drop='first')
    enc_df = pd.DataFrame(enc.fit_transform(df[['occupation']]).toarray())
    df = df.join(enc_df)
    df = df.reset_index(drop=True)
    def aging(x):
        if 7 <= x <= 15:
            return 0
        elif 15 < x <= 25:
            return 1
        elif 25 < x <= 35:
            return 2
        elif 35 < x <= 45:
            return 3
        elif 45 < x <= 55:
            return 4
        elif 55 < x <= 65:
            return 5
        elif 65 < x <= 75:
            return 6
    df['age'] = df['age'].apply(aging)
    enc = OneHotEncoder(drop='first')
    enc_df = pd.DataFrame(enc.fit_transform(df[['age']]).toarray())
    enc_df.columns = ['age0', 'age1', 'age2', 'age3', 'age4', 'age5']
    df = df.join(enc_df)
    df.drop(['occupation'], axis=1, inplace=True)
    df.drop(['age'], axis=1, inplace=True)
    Total = list(itertools.chain.from_iterable(itertools.repeat(counts[x], counts[x]) for x in range(len(counts))))
    ToTal = pd.DataFrame(Total, columns=['Total'])
    DaTa = pd.concat([df, ToTal], axis=1)
    for t in range(len(DaTa['time'])):
        DaTa['time'][t] = datetime.datetime.fromtimestamp(DaTa['time'][t])
    DaTa['time'] = DaTa['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
    DaTa['time'] = DaTa['time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())
    max_time = DaTa['time'].max()
    min_time = DaTa['time'].min()
    delta = max_time - min_time
    Date_range = pd.date_range(min_time, max_time, freq='d')
    df_DATE1 = pd.DataFrame(Date_range, columns=['time'])
    df_DATE1['time'] = df_DATE1['time'].apply(lambda x: x.to_pydatetime().date())
    Date_Integer = np.arange(0, len(Date_range), 1, dtype=int)
    df_DATE2 = pd.DataFrame(Date_Integer, columns=['Integer Date'])
    df_DATE = pd.concat([df_DATE1, df_DATE2], axis=1)
    New_Dataset_F = pd.merge(DaTa, df_DATE, on='time')
    New_Dataset_F = New_Dataset_F.sort_values(by=['FileID', 'Integer Date'])
    New_Dataset_ = New_Dataset_F.copy()
    New_Dataset_.drop(['time'], axis=1, inplace=True)
    New_Dataset_['survival time'] = New_Dataset_['Integer Date']
    mint_time = 60
    Number_of_Iteration = 140
    Time_Threshold = 20
    values, counts = np.unique(New_Dataset_['FileID'], return_counts=True)
    count1 = 0
    count0 = 0
    Number_Patients = len(counts)
    New_Dataset_['label'] = New_Dataset_['Total']
    del New_Dataset_['Total']
    total_count = 0
    x = np.zeros((Number_Patients, Time_Threshold, New_Dataset_.shape[1]))
    XX = np.zeros((Number_Patients * (Number_of_Iteration - mint_time), Time_Threshold, New_Dataset_.shape[1]))
    last_meas = np.zeros(Number_Patients * (Number_of_Iteration - mint_time))
    time = np.zeros(Number_Patients * (Number_of_Iteration - mint_time))
    label = np.zeros(Number_Patients * (Number_of_Iteration - mint_time))
    round_ = 0
    for Current_time in range(mint_time, Number_of_Iteration):
        if Current_time !=  mint_time:
            break
        else:
            count_id = 0
            for id_ in sorted(list(set(DaTa['FileID']))):
                Temp = New_Dataset_[New_Dataset_['FileID'] == id_]
                total_count = total_count + 1
                if Current_time in set(Temp['Integer Date']):
                    index = list(np.where(Temp['Integer Date'] == Current_time))
                    index = index[-1][0]
                    if index >= Time_Threshold:
                        x[count_id, :, :] = Temp[index - Time_Threshold:index]
                        x[count_id, :, -2] = x[count_id, -1, -3]
                        if (len(Temp[Temp['Integer Date'] == (Current_time + 1)]) != 0) | (len(Temp[Temp['Integer Date'] == (Current_time + 2)]) != 0) | (len(Temp[Temp['Integer Date'] == (Current_time + 3)]) != 0)  | (len(Temp[Temp['Integer Date'] == (Current_time + 4)]) != 0):#| (len(Temp[Temp['Integer Date'] == (Current_time + 5)]) != 0):
                            x[count_id, :, -1] = 1
                            label[count_id + Number_Patients * (round_)] = 1
                            count1 = count1 + 1
                            last_meas[count_id + Number_Patients * (round_)] = x[count_id, -1, -3]
                            time[count_id + Number_Patients * (round_)] = x[count_id, -1, -2]
                        else:
                            x[count_id, :, -1] = 0
                            label[count_id + Number_Patients * (round_)] = 0
                            count0 = count0 + 1
                            last_meas[count_id + Number_Patients * (round_)] = x[count_id, -1, -3]
                            time[count_id + Number_Patients * (round_)] = x[count_id, -1, -2]
                        count_id = count_id + 1
                    else:
                        x[count_id, Time_Threshold - index:, :] = Temp[:index]
                        x[count_id, :Time_Threshold - index - 1, :] = np.zeros((Time_Threshold - index - 1, New_Dataset_.shape[1]))
                        x[count_id, :, -2] = x[count_id, -1, -3]
                        if (len(Temp[Temp['Integer Date'] == (Current_time + 1)]) != 0) | (len(Temp[Temp['Integer Date'] == (Current_time + 2)]) != 0) | (len(Temp[Temp['Integer Date'] == (Current_time + 3)]) != 0)  | (len(Temp[Temp['Integer Date'] == (Current_time + 4)]) != 0):#| (len(Temp[Temp['Integer Date'] == (Current_time + 5)]) != 0):
                            x[count_id, :, -1] = 1
                            label[count_id + Number_Patients * (round_)] = 1
                            count1 = count1 + 1
                            last_meas[count_id + Number_Patients * (round_)] = x[count_id, -1, -3]
                            time[count_id + Number_Patients * (round_)] = x[count_id, -1, -2]
                        else:
                            x[count_id, :, -1] = 0
                            label[count_id + Number_Patients * (round_)] = 0
                            count0 = count0 + 1
                            last_meas[count_id + Number_Patients * (round_)] = x[count_id, -1, -3]
                            time[count_id + Number_Patients * (round_)] = x[count_id, -1, -2]
                        count_id = count_id + 1
                else:
                    Temp_2 = Temp[Temp['Integer Date'] <= Current_time]
                    if np.shape(Temp_2)[0] == Time_Threshold:
                        x[count_id, :, :] = Temp_2
                        x[count_id, :, -2] = x[count_id, -1, -3]
                        if (len(Temp[Temp['Integer Date'] == (Current_time + 1)]) != 0) | (len(Temp[Temp['Integer Date'] == (Current_time + 2)]) != 0) | (len(Temp[Temp['Integer Date'] == (Current_time + 3)]) != 0)  | (len(Temp[Temp['Integer Date'] == (Current_time + 4)]) != 0):#| (len(Temp[Temp['Integer Date'] == (Current_time + 5)]) != 0):
                            x[count_id, :, -1] = 1
                            label[count_id + Number_Patients * (round_)] = 1
                            count1 = count1 + 1
                            last_meas[count_id + Number_Patients * (round_)] = x[count_id, -1, -3]
                            time[count_id + Number_Patients * (round_)] = x[count_id, -1, -2]
                        else:
                            x[count_id, :, -1] = 0
                            label[count_id + Number_Patients * (round_)] = 0
                            count0 = count0 + 1
                            last_meas[count_id + Number_Patients * (round_)] = x[count_id, -1, -3]
                            time[count_id + Number_Patients * (round_)] = x[count_id, -1, -2]
                        count_id = count_id + 1
                    elif np.shape(Temp_2)[0] > Time_Threshold:
                        x[count_id, :, :] = Temp_2[-Time_Threshold:]
                        x[count_id, :, -2] = x[count_id, -1, -3]
                        if (len(Temp[Temp['Integer Date'] == (Current_time + 1)]) != 0) | (len(Temp[Temp['Integer Date'] == (Current_time + 2)]) != 0) | (len(Temp[Temp['Integer Date'] == (Current_time + 3)]) != 0)  | (len(Temp[Temp['Integer Date'] == (Current_time + 4)]) != 0):#| (len(Temp[Temp['Integer Date'] == (Current_time + 5)]) != 0):
                            x[count_id, :, -1] = 1
                            label[count_id + Number_Patients * (round_)] = 1
                            count1 = count1 + 1
                            last_meas[count_id + Number_Patients * (round_)] = x[count_id, -1, -3]
                            time[count_id + Number_Patients * (round_)] = x[count_id, -1, -2]
                        else:
                            x[count_id, :, -1] = 0
                            label[count_id + Number_Patients * (round_)] = 0
                            count0 = count0 + 1
                            last_meas[count_id + Number_Patients * (round_)] = x[count_id, -1, -3]
                            time[count_id + Number_Patients * (round_)] = x[count_id, -1, -2]
                        count_id = count_id + 1
                    elif (np.shape(Temp_2)[0] < Time_Threshold) & (np.shape(Temp_2)[0] > 0) :
                        x[count_id, Time_Threshold - np.shape(Temp_2)[0]:, :] = Temp_2
                        x[count_id, :Time_Threshold - np.shape(Temp_2)[0] - 1, :] = np.zeros(
                            (Time_Threshold - np.shape(Temp_2)[0] - 1, New_Dataset_.shape[1]))
                        x[count_id, :, -2] = x[count_id, -1, -3]
                        if (len(Temp[Temp['Integer Date'] == (Current_time + 1)]) != 0) | (len(Temp[Temp['Integer Date'] == (Current_time + 2)]) != 0) | (len(Temp[Temp['Integer Date'] == (Current_time + 3)]) != 0)  | (len(Temp[Temp['Integer Date'] == (Current_time + 4)]) != 0):#| (len(Temp[Temp['Integer Date'] == (Current_time + 5)]) != 0):
                            x[count_id, :, -1] = 1
                            label[count_id + Number_Patients * (round_)] = 1
                            count1 = count1 + 1
                            last_meas[count_id + Number_Patients * (round_)] = x[count_id, -1, -3]
                            time[count_id + Number_Patients * (round_)] = x[count_id, -1, -2]
                        else:
                            x[count_id, :, -1] = 0
                            label[count_id + Number_Patients * (round_)] = 0
                            count0 = count0 + 1
                            last_meas[count_id + Number_Patients * (round_)] = x[count_id, -1, -3]
                            time[count_id + Number_Patients * (round_)] = x[count_id, -1, -2]
                        count_id = count_id + 1
                    else:
                        count_id = count_id + 1
            XX[Number_Patients * round_:Number_Patients * (round_ + 1), :, :] = x
            round_ = round_ + 1
    ZZ = np.zeros((Number_Patients * (Number_of_Iteration - mint_time), Time_Threshold, New_Dataset_.shape[1]))
    label_ =np.zeros(Number_Patients * (Number_of_Iteration - mint_time))
    last_meas_ = np.zeros(Number_Patients * (Number_of_Iteration - mint_time))
    time_ = np.zeros(Number_Patients * (Number_of_Iteration - mint_time))
    c = 0
    for i in range(np.shape(XX)[0]):
        if len(np.where(~XX[i,:,:].any(axis=1))[0]) != Time_Threshold:
            ZZ[c,:,:] = XX[i, :, :]
            label_[c] = label[i]
            last_meas_[c] = last_meas[i]
            time_[c] = time[i]
            c = c + 1
    ZZ = ZZ[:c, : , :]
    label_ = label_[:c]
    last_meas_ =last_meas_[:c]
    time_ = time_[:c]
    X = np.reshape(ZZ, (c * Time_Threshold, -1))
    New_Dataset = pd.DataFrame(X, columns=New_Dataset_.columns)
    class_count_0, class_count_1 = New_Dataset['label'].value_counts()
    class_0 = New_Dataset[New_Dataset['label'] == 0]
    class_1 = New_Dataset[New_Dataset['label'] == 1]
    print('class 0:', class_0.shape)
    print('class 1:', class_1.shape)
    class_1_under = class_1[:class_0.shape[0]]
    New_Dataset_New = pd.concat([class_1_under, class_0], axis=0)
    Scale = New_Dataset_New.copy()
    Scale.drop(['user_id', 'FileID', 'Integer Date', 'survival time', 'label'], axis=1, inplace=True)
    scaler = MinMaxScaler()
    Scale_Dataset = scaler.fit_transform(Scale)
    Scale_Dataset = pd.DataFrame(Scale_Dataset, columns=Scale.columns)
    DF_Time = New_Dataset_New['Integer Date']
    DF_Time = DF_Time.reset_index(drop=True)
    DF_Survival = New_Dataset_New['survival time']
    DF_Survival = DF_Survival.reset_index(drop=True)
    DF_Label = New_Dataset_New['label']
    DF_Label = pd.DataFrame(DF_Label, columns=['label'])
    DF_Label = DF_Label.reset_index(drop=True)
    data = pd.concat([Scale_Dataset, DF_Time, DF_Survival, DF_Label], axis=1)
    names = data.columns.tolist()
    names[names.index('Integer Date')] = 'times'
    names[names.index('survival time')] = 'survival_time'
    data.columns = names
    num_category = 200
    num_event = len(np.unique(label)) - 1
    mask1 = f_get_fc_mask1(last_meas_, num_event, num_category)
    mask2 = f_get_fc_mask2(time_, label_, num_event, num_category)
    mask3 = f_get_fc_mask3(time_, -1, num_category)
    mask1, mask2, mask3 = mask_divide(label_,mask1,mask2,mask3)
    data_0 = np.asarray(data[data['label'] == 0])
    data_1 = np.asarray(data[data['label'] == 1])
    print('number of 0 label: ', len(data_0))
    print('number of 1 label: ', len(data_1))
    feature_dims = data.shape[1] - 3
    hidden_size = 128
    visit_len = 20
    features_0, labels_0, days_0, ett_0 = mimic_data_divided(data_0, visit_len, feature_dims)
    features_1, labels_1, days_1, ett_1 = mimic_data_divided(data_1, visit_len, feature_dims)
    features = [features_0, features_1]
    labels = [labels_0, labels_1]
    days = [days_0, days_1]
    ett = [ett_0, ett_1]
    model_type = 'train_model'
    l2_regularization = 0.000001
    learning_rate = 0.001
    print(model_type)
    for i in range(train_repeat):
        print("iteration number: %d" % i)
        k_folds = 5
        test_size = 0.2
        train_size = 1 - test_size
        split = mimic_data_spilt(features, labels, 2)
        for ith_fold in range(k_folds):
            print('{} th fold of {} folds'.format(ith_fold, k_folds))
            train_set, test_set = mimic_data_concat(split, features, labels, days, ett, mask1, mask2, mask3, num_event)
            ac , pre0, pre1, rec0, rec1, sco0, sco1,  encoder , FC, decoder, mlp= train_model(
                    train_set=train_set,
                    test_set=test_set,
                    feature_dims=feature_dims,
                    hidden_size=hidden_size,
                    num_category=num_category,
                    num_event=num_event,
                    learning_rate=learning_rate,
                    l2_regularization=l2_regularization,
                    MASK_RATE=MASK_RATE,
                    SHUFFLE_RATE=SHUFFLE_RATE,
                    ith_fold = ith_fold
            )

            rec1_list.append(rec1)
            rec0_list.append(rec0)
            sco1_list.append(sco1)
            sco0_list.append(sco0)
            pre1_list.append(pre1)
            pre0_list.append(pre0)
            ac_list.append(ac)
        pre0_List_total[i,:]= pre0_list
        pre1_List_total[i,:]= pre1_list
        rec0_List_total[i,:]= rec0_list
        rec1_List_total[i,:]= rec1_list
        sco0_List_total[i,:]= sco0_list
        sco1_List_total[i,:]= sco1_list
        ac_List_total[i,:]= ac_list
        print('epoch  {}-----accuracy all_ave  {}'.format(i, np.mean(ac_list, axis=0)))
        ac_list = []
        pre0_list = []
        pre1_list = []
        rec0_list = []
        rec1_list = []
        sco0_list = []
        sco1_list = []
        pre0_list__ = pd.DataFrame(pre0_List_total, columns=['precision_fold1','precision_fold2','precision_fold3', 'precision_fold4', 'precision_fold5'])
        pre1_list__ = pd.DataFrame(pre1_List_total, columns=['precision_fold1','precision_fold2','precision_fold3', 'precision_fold4', 'precision_fold5'])
        pre0_list__.to_excel('result//Movilens_Precision_0_{}_lr={}_3_mask_rate={}_srate={}.xlsx'.format(model_type, learning_rate, MASK_RATE, SHUFFLE_RATE), index=False)
        pre1_list__.to_excel('result//Movilens_Precision_1_{}_lr={}_3_mask_rate={}_srate={}.xlsx'.format(model_type, learning_rate, MASK_RATE, SHUFFLE_RATE), index=False)

        rec0_list__ = pd.DataFrame(rec0_List_total, columns=['recall_fold1','recall_fold2','recall_fold3', 'recall_fold4', 'recall_fold5'])
        rec1_list__ = pd.DataFrame(rec1_List_total, columns=['recall_fold1','recall_fold2','recall_fold3', 'recall_fold4', 'recall_fold5'])
        rec0_list__.to_excel('result//Movilens_Recall_0_{}_lr={}_3_mask_rate={}_srate={}.xlsx'.format(model_type, learning_rate, MASK_RATE, SHUFFLE_RATE), index=False)
        rec1_list__.to_excel('result//Movilens_Recall_1_{}_lr={}_3_mask_rate={}_srate={}.xlsx'.format(model_type, learning_rate, MASK_RATE, SHUFFLE_RATE), index=False)

        sco0_list__ = pd.DataFrame(sco0_List_total, columns=['f1-score_fold1','f1-score_fold2','f1-score_fold3', 'f1-score_fold4', 'f1-score_fold5'])
        sco1_list__ = pd.DataFrame(sco1_List_total, columns=['f1-score_fold1','f1-score_fold2','f1-score_fold3', 'f1-score_fold4', 'f1-score_fold5'])
        sco0_list__.to_excel('result//Movilens_Score_0_{}_lr={}_3_mask_rate={}_srate={}.xlsx'.format(model_type, learning_rate, MASK_RATE, SHUFFLE_RATE), index=False)
        sco1_list__.to_excel('result//Movilens_Score_1_{}_lr={}_3_mask_rate={}_srate={}.xlsx'.format(model_type, learning_rate, MASK_RATE, SHUFFLE_RATE), index=False)

        ac_list__ = pd.DataFrame(ac_List_total, columns=['accuracy_fold1','accuracy_fold2','accuracy_fold3', 'accuracy_fold4', 'accuracy_fold5'])
        ac_list__.to_excel('result//Movilens_acc_{}_lr={}_3_mask_rate={}_srate={}.xlsx'.format(model_type, learning_rate, MASK_RATE, SHUFFLE_RATE), index=False)
        print('param1', encoder.summary())
        print('param3', decoder.summary())
        print('param4', mlp.summary())
if __name__ == '__main__':
    experiment(0, 1)

