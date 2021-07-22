

def change_data_size(cfg,data):
    new_data = data.view(cfg.batch_size_test,cfg.height,cfg.width,cfg.channel)
    return new_data

def rechange_data_size(cfg,data):
    new_data = data.view(cfg.batch_size_test,cfg.channel,cfg.height,cfg.width,)
    return new_data