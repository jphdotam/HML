import os
import yaml

def load_config(configpath):
    with open(configpath) as f:
        cfg = yaml.safe_load(f)

    experiment_id = os.path.splitext(os.path.basename(configpath))[0]
    cfg['experiment_id'] = experiment_id

    vis_dir = os.path.join("./output/vis", experiment_id)
    model_dir = os.path.join("./output/models", experiment_id)
    log_dir = os.path.join("./output/logs", experiment_id)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return cfg, model_dir, vis_dir, log_dir