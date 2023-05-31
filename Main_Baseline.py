import argparse
import os
import pandas as pd
from pathlib import Path
import time
import datetime

from Helpers.models.Networks.Model_DeepConvNet import DeepConvNet
from Helpers.models.Networks.Model_EEGNet4 import EEGNet4
from Helpers.models.Networks.Model_EEGNet8 import EEGNet8
from Helpers.models.Networks.Model_ResNet1D18 import Resnet18
from Helpers.models.Networks.Model_ResNet1D8 import Resnet8

from Helpers.trainer import Trainer
from Helpers.Variables import device, METRICS, WD, FILENAME_TOTALSUM, FILENAME_TOTALRES, FILENAME_MODEL, FILENAME_FOLDSUM
 
"""
사용할 데이터셋 선택 
1. LOSO_BIOCAN
2. LOSO_BIO
3. dependent_BIOCAN
4. dependent_BIO
"""
from Helpers.Helpers_LOSO_BIOCAN import data_generate, BIODataset, BIODataLoader, pathfinder 
# from Helpers.Helpers_LOSO import data_generate, BIODataset, BIODataLoader, pathfinder 
# from Helpers.Helpers_BIOCAN import data_generate, BIODataset, BIODataLoader, pathfinder
# from Helpers.Helpers import data_generate, BIODataset, BIODataLoader, pathfinder 


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def Experiment(args):
    RESD = os.path.join(WD, 'res') # 결과 저장 폴더 경로
    data_flen = len(os.listdir(f'{WD}/data')) # 가장 최근 생성한 데이터 선택

    # 결과 저장 경로 설정 1
    Path(RESD).mkdir(parents=True, exist_ok=True)
    res_flen = str(len(os.listdir(RESD)))
    print(f"Saving results to res/{res_flen}")
    
    ts_fold = pd.DataFrame(columns=METRICS)
    # k-fold
    for nf in range(1, 6):
        ts_total = pd.DataFrame(columns=METRICS)
        
        print('='*30)
        print(' '*11, 'FOLD', nf)
        print('='*30)

        # processed data 불러올 경로 설정
        data_load_dir = f'{WD}/data/{data_flen-1}/{nf}fold'
        print(f'Loaded data from data/{data_flen-1}/fold{nf}')   

        nfoldname = f'fold{nf}'
        fold_dir = os.path.join(RESD, res_flen, nfoldname)

        for subj in range(1, 24):
            for expt in range(1, 3):
                print('='*30)
                print(' '*4, '피험자{} - 실험{} 시작'.format(subj, expt))
                print('='*30)
                # 데이터 정보 불러오기
                _, _, sbj = pathfinder(subj, expt)

                # 결과 저장 경로 설정 2
                res_name = f'S{sbj}_EXPT{expt}'
                res_dir = os.path.join(RESD, res_flen, nfoldname, res_name)
                Path(res_dir).mkdir(parents=True, exist_ok=True)

                tr_dataset = BIODataset('train', device, data_load_dir, res_name)
                train_loader = BIODataLoader(dataset=tr_dataset, batch_size=int(args.BATCH), \
                                                num_workers=0, shuffle=False, drop_last=True)
                vl_dataset = BIODataset('valid', device, data_load_dir, res_name)
                valid_loader = BIODataLoader(dataset=vl_dataset, batch_size=int(args.BATCH), \
                                                num_workers=0, shuffle=False, drop_last=True)
                ts_dataset = BIODataset('test', device, data_load_dir, res_name)
                test_loader = BIODataLoader(dataset=ts_dataset, batch_size=int(args.BATCH), \
                                                num_workers=0, shuffle=False, drop_last=True)
                """
                Baseline model selection
                This models are only trained by EEG data (uni-modality)
                """
                ###### 모델 생성
                # my_model = EEGNet4(args.n_classes, args.n_channels, args.freq_time, True).to(device)
                # my_model = EEGNet8(args.n_classes, args.n_channels, args.freq_time, True).to(device)
                my_model = DeepConvNet(args.n_classes, args.n_channels, args.freq_time, True).to(device)
                # my_model = Resnet8
                MODEL_PATH = os.path.join(res_dir, FILENAME_MODEL)
                MODEL_PATH = f'{MODEL_PATH}'

                # 학습
                trainer = Trainer(args, my_model, MODEL_PATH) 
                tr_history = trainer.train(train_loader, valid_loader)
                print('End of Train\n')

                # Test set 성능 평가
                ts_history = trainer.eval('test', test_loader)
                print('End of Test\n')
                
                trainer.writer.close()

                # Save Results
                trainer.save_result(tr_history, ts_history, res_dir)

                ts_total = pd.concat([ts_total, ts_history], axis=0, ignore_index=True)

        ts_total.to_csv(os.path.join(fold_dir, FILENAME_TOTALRES))
        ts_total.describe().to_csv(os.path.join(fold_dir, FILENAME_TOTALSUM))

        ts_fold = pd.concat([ts_fold, ts_total], axis=0, ignore_index=True)
    ts_fold.describe().to_csv(os.path.join(RESD, res_flen, FILENAME_FOLDSUM))

if __name__ == '__main__':
    start = time.time()
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--with_CAN', default=True, help='When you want to use CAN data, setting to "True"')
    parser.add_argument('--data_type', default='independent', help='Choose dataset type for dependent and independent(LOSO) experiments')

    parser.add_argument('--BATCH', default=16, help='Batch Size') # original 16
    parser.add_argument('--EPOCH', default=100, help='Epoch') # original: 50, set: 10
    parser.add_argument('--optimizer', default='Adam', help='Optimizer')
    parser.add_argument('--lr', default=0.002, help='Adam Learning Rate') # original: 1e-2, set: 2e-2
    parser.add_argument('--freq_time', default=750, help='frequency(250)*time window(3)')
    parser.add_argument('--n_channels', default=28)
    parser.add_argument('--n_classes', default=3)
    args = parser.parse_args()
    
    # Data Generation at first time
    if not os.path.exists(os.path.join(WD, f'data/{args.data_type}')):
        data_generate()
    
    Experiment(args) # Time -> about 1 hour 50 minutes
    print('Code Time Consumption: ', str(datetime.timedelta(seconds=time.time() - start)).split('.')[0])

