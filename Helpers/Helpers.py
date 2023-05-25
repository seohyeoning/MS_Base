import os
from tqdm import tqdm
from pathlib import Path
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from Helpers.Variables import device, WS, WD, COL_NAMES, DATA_DIR, CLASS_NUM

class BIODataset(Dataset):
    def __init__(self, phase, device, data_load_dir, res_name):
        super().__init__()
        self.device = device
        
        self.data = np.load(f'{data_load_dir}/{res_name}_{phase}.npz')
        self.X = self.data['data']
        self.y = self.data['label']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx]).to(self.device)
        y = torch.FloatTensor(self.y[idx]).to(self.device)
        return x, y
    

class BIODataLoader(DataLoader): # 왜 끊긴 것 같지?
    def __init__(self, *args, **kwargs):
        super(BIODataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
    
def _collate_fn(batch): # 배치사이즈 지정 방법 확인
    x_batch, y_batch = [], torch.Tensor().to(device)
    xe_batch, xc_batch, xr_batch, xp_batch, xg_batch = torch.Tensor().to(device), torch.Tensor().to(device), \
                                                       torch.Tensor().to(device), torch.Tensor().to(device), \
                                                       torch.Tensor().to(device)
    for (_x, _y) in batch:
        # 1. 데이터(x)에서 EEG와 나머지 분리하기
        # 2. 데이터 shape 3차원으로 맞춰주기
        # 3. numpy -> tensor
        xe = _x[:, :-4]                        # EEG
        xc = torch.unsqueeze(_x[:, -4], 1)     # ECG
        xr = torch.unsqueeze(_x[:, -3], 1)     # Resp
        xp = torch.unsqueeze(_x[:, -2], 1)     # PPG
        xg = torch.unsqueeze(_x[:, -1], 1)     # GSR    

        # Dimension swap: (N, Seq, Ch) -> (N, Ch, Seq)
        xe = torch.permute((xe), (1, 0)).to(dtype=torch.float32)
        xc = torch.permute((xc), (1, 0)).to(dtype=torch.float32)
        xr = torch.permute((xr), (1, 0)).to(dtype=torch.float32)
        xp = torch.permute((xp), (1, 0)).to(dtype=torch.float32)
        xg = torch.permute((xg), (1, 0)).to(dtype=torch.float32)

        xe = torch.unsqueeze(xe, 0) # (28, sr*sec) -> (1, 28, sr*sec)
        xc = torch.unsqueeze(xc, 0) # (1, sr*sec) -> (1, 1, sr*sec)
        xr = torch.unsqueeze(xr, 0)
        xp = torch.unsqueeze(xp, 0)
        xg = torch.unsqueeze(xg, 0)

        xe_batch = torch.cat((xe_batch, xe), 0)
        xc_batch = torch.cat((xc_batch, xc), 0)
        xr_batch = torch.cat((xr_batch, xr), 0)
        xp_batch = torch.cat((xp_batch, xp), 0)
        xg_batch = torch.cat((xg_batch, xg), 0)

        _y = torch.unsqueeze(_y, 0)

        y_batch = torch.cat((y_batch, _y), 0) # (3, ) -> (1, 3)    

        
    x_batch = [xe_batch, xc_batch, xr_batch, xp_batch, xg_batch]
    
    return {'data': x_batch, 'label': y_batch}

########## 데이터 파일 경로 지정 함수
def pathfinder(s, e):
    if s < 10:
        s = '0' + str(s)
    names = [(f'S{s}_DAY{d}_EXPT{e}_DRIVE_preprocess.csv') for d in range(1, 3)]
    paths = [(os.path.join(DATA_DIR, n)) for n in names]
    
    return paths[0], paths[1], str(s)

########## 데이터 불러와서 정리하는 함수
def import_data(path):
    # 원본 데이터 가져오기
    raw_data = pd.read_csv(path)
    raw_data = raw_data.iloc[::2]     # downsampling 500 -> 250Hz
    
    # features와 labels 나누기
    raw_features = raw_data.iloc[:, 1:-2]
    raw_labels = pd.DataFrame(raw_data.iloc[:, -1], columns=['TRIGGER(DIGITAL)'])

    '''
    설문 내용에 따라 재분류
    0 : no = 0
    1~3 : mild = 1
    4~6 : moderate = 2 
    '''

    # 레이블 재구성
    old_labels = list(raw_labels['TRIGGER(DIGITAL)'])
    new_labels = [0 if i == 0 else 1 if i < 4 else 2 for i in old_labels]
    labels = pd.DataFrame(new_labels)

    scaler = MinMaxScaler()

    # features 데이터 정규화
    features = pd.DataFrame(scaler.fit_transform(raw_features))
    features.columns = COL_NAMES
    
    return features, labels   

 ########## Window segmentation 함수
def segment(data, label, window_size=750, step_=750): # no overlapping
    feature_list = []
    label_list = []
    
    print('feature length: \t', len(data), 'window size: ', window_size)
    for i in range(0, len(data) - window_size + 1, int(step_)):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(stats.mode(label.iloc[i:i+window_size])[0][0][0])
    return np.array(feature_list), np.array(label_list)

def data_generate():
    STEP = WS
    
    # processed data 저장 경로 설정
    Path(f'{WD}/data').mkdir(parents=True, exist_ok=True)
    data_fnum = len(os.listdir(f'{WD}/data'))
    data_save_dir = f'{WD}/data/{data_fnum}'
    Path(data_save_dir).mkdir(parents=True, exist_ok=True)

    """
    순서: 한 사람마다 expt1+2하고, 5-fold로 train+val/test 나눔
          각자 나눠준 뒤, 각 사람들의 train/val/test npz 저장.

          나중에 불러와서 dependent로 함. 사람들 data 더하고싶은만큼 더해도 되고, 한명만 해도 되고!
    """
    for subj in range(1, 24):
        for expt in tqdm(range(1, 3)):

            # 데이터 정보 불러오기
            path_day1, path_day2, sbj = pathfinder(subj, expt)
            res_name = f'S{sbj}_EXPT{expt}'
            
            # raw data 1차 전처리 (feature 정규화, label 재분류)
            features1, labels1 = import_data(path_day1)
            features2, labels2 = import_data(path_day2)
            
            # Window Segmentation
            segmented_features1, segmented_labels1 = segment(features1, labels1, WS, STEP)
            segmented_features2, segmented_labels2 = segment(features2, labels2, WS, STEP)

            # 합치기 (Day1+Day2)
            segmented_features = np.concatenate((segmented_features1, segmented_features2), axis=0)
            segmented_labels = np.concatenate((segmented_labels1, segmented_labels2), axis=0)

            # train, validation, test set 나누기
            kf = KFold(n_splits=5, shuffle=True, random_state=0) # 5 fold
            kf_gen = kf.split(segmented_features)

            cnt = 0
            for tr_idx, ts_idx in kf_gen:
                cnt += 1
                final_datasave_dir = os.path.join(data_save_dir, f'{cnt}fold')
                Path(final_datasave_dir).mkdir(parents=True, exist_ok=True)

                # nth-fold 내부에서 train, validation, test set 나누기
                x_0, x_test = segmented_features[tr_idx], segmented_features[ts_idx]
                y_0, y_test = segmented_labels[tr_idx], segmented_labels[ts_idx]
                x_train, x_valid, y_train, y_valid = train_test_split(
                    x_0, 
                    y_0, 
                    test_size = 0.3,
                    random_state = 0
                ) 
 
                # 레이블 원핫 인코딩
                onehot_y_train = to_categorical(y_train, num_classes=CLASS_NUM)
                onehot_y_valid = to_categorical(y_valid, num_classes=CLASS_NUM)
                onehot_y_test = to_categorical(y_test, num_classes=CLASS_NUM)       

                # 저장     
                fname_train = f"{res_name}_train.npz"
                fname_valid = f"{res_name}_valid.npz"
                fname_test = f"{res_name}_test.npz"

                np.savez(f"{final_datasave_dir}/{fname_train}", data=x_train, label=onehot_y_train)
                np.savez(f"{final_datasave_dir}/{fname_valid}", data=x_valid, label=onehot_y_valid)
                np.savez(f"{final_datasave_dir}/{fname_test}", data=x_test, label=onehot_y_test)

            print(f'Done processing {res_name}.')

            
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]