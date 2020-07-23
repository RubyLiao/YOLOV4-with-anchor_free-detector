class CONFIG:
    pretrained = False
    pretrained_path = ''
    start_epoch = 13
    GPU_device = "0"
    log_dir = './logs'
    base_train_data = r'./data.txt'

    base_gt_json = r'./gt.json'
    base_name_id_json = r'./name_id.json'

    cls_num = 18
    stride = 4
    if_giou = False
    train_batch_size = 1
    eval_iter = 6000
    eval_epoch = 1

    input_size = (480, 640)

    lr = {0: 0.001, 4: 0.0005, 8: 0.0001}
    epoch_num = 10

    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]

    model_dir = ''

    base_num = 1000
    add_num = 300
    weights = [0.5, 2.0]

    cls_id = {
        'cls_name': 0
    }