import argparse
import os
import pandas as pd
from pathlib import Path
import time
import datetime
from Helpers.models.Base_Model import AMFTE
from Helpers.trainer import Trainer
from Helpers.Helpers import data_generate, BIODataset, BIODataLoader, pathfinder 
from Helpers.Variables import device, METRICS, WD, FILENAME_TOTALSUM, FILENAME_TOTALRES, FILENAME_MODEL, FILENAME_FOLDSUM
 
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
                                                num_workers=0, shuffle=True, drop_last=True)
                vl_dataset = BIODataset('valid', device, data_load_dir, res_name)
                valid_loader = BIODataLoader(dataset=vl_dataset, batch_size=int(args.BATCH), \
                                                num_workers=0, shuffle=True, drop_last=True)
                ts_dataset = BIODataset('test', device, data_load_dir, res_name)
                test_loader = BIODataLoader(dataset=ts_dataset, batch_size=int(args.BATCH), \
                                                num_workers=0, shuffle=True, drop_last=True)
                
                ###### 모델 생성
                # my_model = Net().to(device) 
                # my_model = AMFTE().to(device)
                my_model = AdaNet().to(device)
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
    parser.add_argument('--BATCH', default=16, help='Batch Size') # original 16
    parser.add_argument('--EPOCH', default=100, help='Epoch') # original: 50, set: 10
    parser.add_argument('--optimizer', default='Adam', help='Optimizer')
    parser.add_argument('--lr', default=0.002, help='Adam Learning Rate') # original: 1e-2, set: 2e-2

    args = parser.parse_args()
    
    # Data Generation at first time
    if not os.path.exists(os.path.join(WD, 'data')):
        data_generate()
    
    Experiment(args) # Time -> about 1 hour 50 minutes
    print('Code Time Consumption: ', str(datetime.timedelta(seconds=time.time() - start)).split('.')[0])

