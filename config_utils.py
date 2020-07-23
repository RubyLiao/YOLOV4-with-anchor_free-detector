
import os


#1w+300训练 1000+300测试
#15个epoch lr=[0.0001， 0.00005, 0.000005]
#bs=16, 约6000次迭代/epoch
#每训练完一个add, 看它在base测试集上有没有变差？在add测试集上有没有变好？ 因为我们希望的是它在add测试集上变好的同时，在base测试集上不变差



class CONFIG:
    UPDATE_EVAL_DATASET = True
    pretrained = False
    pretrained_path = r'D:\Projects-for-study\centerNet-ljr\models_save\chemistry.pth'
    start_epoch = 13
    GPU_device = "0"
    log_dir = './logs'
    base_train_data = r'D:\Data\CMS01_single-end\all_data.txt'
    # base_train_data = r'D:\Data\centernet-ly\pics\dataset-labeled\train\data.txt'
    # base_train_data = r'D:\Projects-for-study\CenterNet-djw\my_data\train\data.txt'

    base_test_data = r'D:\Data\tianping-ljr-laboratory\v3-val\gt-front\val_data.txt'
    base_gt_json = r'D:\Data\tianping-ljr-laboratory\v3-val\gt-front\gt.json'
    base_name_id_json = r'D:\Data\tianping-ljr-laboratory\v3-val\gt-front\name_id.json'

    # base_test_data = r'D:\Data\centernet-ly\pics\dataset-labeled\two\val_two_data.txt'
    # base_gt_json = r'D:\Data\centernet-ly\pics\dataset-labeled\two\gt.json'
    # base_name_id_json = r'D:\Data\centernet-ly\pics\dataset-labeled\two\name_id.json'

    base_pred_save_dir = r'D:\Data\tianping-ljr-laboratory\v3\pred_imgs'


    cls_num = 18
    stride = 4
    if_giou = False
    train_batch_size = 1
    eval_iter = 6000
    eval_epoch = 1

    input_size = (480, 640)

    lr = {0:0.001, 4:0.0005, 8:0.0001}
    epoch_num = 10
    IF_TRAIN_SHOW = False

    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]

    model_dir = r'./models_chem'

    base_num = 1000
    add_num = 300
    weights = [0.5, 2.0]

    cls_id = {
        'power':1,
        'battery':2,
        'switch': 3,
        'ammeter': 4,
        'voltmeter': 5,
        'constant': 6,
        'slide': 7,
        'bulb': 8,
        'lightedBulb': 9,

        'powerPositive': 10,
        'powerNegative': 11,
        'batteryPositive': 12,
        'batteryNegative':13,
        'ammeterNegative': 14,
        'ammeterPositiveMid': 15,
        'ammeterPositiveRight': 16,
        'voltmeterNegative': 17,
        'voltmeterPositiveMid': 18,
        'voltmeterPositiveRight': 19,
        'constantPositive': 20,
        'slidePositive': 21,
        'slideNegative': 22,
        'slideBlock': 23,
        'switchHead': 24,
        'switchPos': 25,
        'switchNeg': 26,
        'bulbPos': 27,
        'bulbNeg': 28
    }

    cls_id2 = {
        'opticalBench': 0,
        'convex':1,
        'screen':2,
        'candle':3,
        'hand':4,
        'shubiao':5
    }
    cls_id3 = {'opticalBench': 0}

    cls_id4 = {
    'liangtong':0,
    'jiaotoudiguan':1,
    'shaobei':2,
    'bolibang':3,
    'loudou':4,
    'lvzhi':5,
    'lvzhiloudou':6,
    'xiping':7,
    'tiejiatai':8,
    'tiejiaquan':9,
    'shiguan':10,
    'shiguankou':11,
    'shiguanwei':12,
    'bolibangduandian':13,
    'jiaotoudiguantou':14,
    'jiaotoudiguanwei':15,
    'loudoujianzui':16,
    'shou':17
    }