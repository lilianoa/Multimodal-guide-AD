import os
import json
import pandas as pd

class VisASolver(object):
    CLSNAMES = [
        'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
        'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
        'pcb4', 'pipe_fryum',
    ]

    def __init__(self, root='data/visa'):
        self.root = root
        self.meta_path = f'{root}/meta_fewshot.json'
        self.phases = ['train', 'test']
        self.csv_data = pd.read_csv(f'{root}/split_csv/2cls_fewshot.csv', header=0)

    def run(self):
        columns = self.csv_data.columns  # [object, split, label, image, mask]
        info = {phase: {} for phase in self.phases}
        anomaly_samples = 0
        normal_samples = 0
        train_anomaly_samples = 0
        train_normal_samples = 0
        test_anomaly_samples = 0
        test_normal_samples = 0
        for cls_name in self.CLSNAMES:
            img_anno = pd.read_csv(f'{self.root}/{cls_name}/image_anno.csv', header=0)
            cls_data = self.csv_data[self.csv_data[columns[0]] == cls_name]
            for phase in self.phases:
                cls_info = []
                cls_data_phase = cls_data[cls_data[columns[1]] == phase]
                cls_data_phase.index = list(range(len(cls_data_phase)))
                for idx in range(cls_data_phase.shape[0]):
                    data = cls_data_phase.loc[idx]
                    is_abnormal = True if data[2] == 'anomaly' else False
                    if is_abnormal:
                        img_path = data[3]
                        img_anno_data = img_anno.loc[img_anno['image'] == img_path, 'label']
                        specie_name = img_anno_data.values[0]
                    else:
                        specie_name = 'normal'
                    info_img = dict(
                        img_path=data[3],
                        mask_path=data[4] if is_abnormal else '',
                        cls_name=cls_name,
                        specie_name=specie_name,
                        anomaly=1 if is_abnormal else 0,
                    )
                    cls_info.append(info_img)
                    if phase == 'test':
                        if is_abnormal:
                            test_anomaly_samples = test_anomaly_samples + 1
                        else:
                            test_normal_samples = test_normal_samples + 1
                    else:
                        if is_abnormal:
                            train_anomaly_samples = train_anomaly_samples + 1
                        else:
                            train_normal_samples = train_normal_samples + 1
                info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        print('test_normal_samples', test_normal_samples, 'test_anomaly_samples', test_anomaly_samples)
        print('train_normal_samples', train_normal_samples, 'train_anomaly_samples', train_anomaly_samples)


if __name__ == '__main__':
    runner = VisASolver(root='./data/visa')
    runner.run()
