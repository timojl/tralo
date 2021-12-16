from os.path import join, realpath, dirname, isfile
import os
from collections import OrderedDict
import numpy as np
import gzip
import json
from string import Template
from flask import Flask, send_from_directory

app = Flask(__name__, static_folder="static")


# base_template = Template(open(join(dirname(realpath(__file__)), 'base.html')).read())

def resample(x, res):
    old_size = len(x)
    return np.interp(np.arange(0, old_size, old_size / res), np.arange(0, old_size), x)

@app.route('/')
def index():
    index = open(join(dirname(realpath(__file__)), 'html/index.html')).read()
    return index


@app.route('/webfonts/<path:filename>')
def base_static(filename):
    return send_from_directory(app.root_path + '/static/webfonts/', filename)


@app.route('/list_logs')
def list_logs():

    folders = []
    for folder in os.listdir('logs'):
        try:

            loss_file = join('logs', folder, 'stats.json')
            if not os.path.isfile(loss_file):
                loss_file = join('logs', folder, 'losses.json')
            # losses = json.load(open(loss_file, 'r'))
            t_create = os.path.getctime(loss_file)
            # if len(losses['iterations']) > 10:
            #     folders += [folder]
            folders += [(folder, t_create)]

        except BaseException:
            pass

    return json.dumps(folders)

    # folders_str = ''
    # for folder in folders:
    #     folders_str += f"""
    #     <label class="panel-block">
    #       <input class="folder-checkbox" type="checkbox" value="{folder}">
    #       {folder}
    #     </label>"""

    # return folders_str

    # return folders_str

    # folders_str = '<div class="select is-multiple"><select multiple size="32">'
    # for folder in folders:
    #     folders_str += f'<option value="{folder}">{folder}</option>'
    # folders_str += '</select></div>'
    # return folders_str
    # return base_template.substitute(content=folders_str)


@app.route('/loss/<query>')
def loss(query):
    print(query)

    try:
        l = json.load(open(join('logs/', query, 'losses.json')))

        if len(l['loss']) > 200:
            l['loss'] = resample(l['loss'], 200)
            l['iterations'] = resample(l['iterations'], 200)
        l['loss'] = [round(x, 3) for x in l['loss']]

        l['loss'] = [-1 if np.isnan(x) else x for x in l['loss']]

        return json.dumps({'loss': list(zip(l['iterations'], l['loss']))})
    except BaseException:
        return json.dumps({'loss': [], 'iterations': []})

@app.route('/statistics/<query>')
def statistics(query):

    data = json.load(open(join('logs/', query, 'stats.json')))

    if all([k in data for k in {'start_time', 'stop_time'}]):
        return json.dumps({'duration': round((data['stop_time'] - data['start_time'])/60, 2)})
    else:
        return json.dumps({})


@app.route('/config/<sub>/<query>')
def config(sub, query):

    expand_keys = {'model_args', 'dataset_args'}
    valid_keys = {
        'general': ['loss', 'model', 'model_args', 'dataset', 'dataset_args', 'config_name', 'interval', 'workers'],
        'opt': ['optimizer',  'batch_size','lr', 'lr_scheduler', 'iterations', 'warmup', 'weight_decay'],
    }
    try:
        config_file = join('logs/', query, 'config.json')

        cfg = json.loads(open(config_file).read()) if isfile(config_file) else {}

       
        valid_keys['other'] = [k for k in cfg if k not in valid_keys['general'] + valid_keys['opt']]

        cfg = OrderedDict([(k, cfg[k]) for k in valid_keys[sub] if k in cfg])

        cfg2 = dict()
        # expand
        for k, v in cfg.items():
            if k not in expand_keys:
                cfg2[k] = v
            else:
                cfg2[k] = '...'
                for k2, v2 in v.items():
                    cfg2['...' + k2] = v2


        # limit length
        cfg2 = {k: v if len(str(v)) < 20 else '..' + str(v)[-20:] for k,v in cfg2.items()}
        
        print(sub, query, cfg2)
        return json.dumps(cfg2)
    except BaseException as e:
        raise e
        return json.dumps({'config': 'not found'})

    # try:
    #     fh = open(join('logs/', query, 'config.json'))
    #     return fh.read()
    # except FileNotFoundError:
    #     return {'config': 'not found'}


@app.route('/utils/<query>')
def all_utilization(query):

    # return json.dumps({'gpu': [1,2,3,4,10,20,30], 'gpu_mem': [1,7,8,9,10,20,30]})
    # try:
    #     return open(join('logs/', query, '.cached-utilization.json')).read()
    # except BaseException:
    #     pass

    try:

        filename = join('logs/', query, 'utilization.json.gz')
        with gzip.open(filename, 'rb') as fh:
            l = json.loads(fh.read())


        # remove keys without data
        l = {k: v for k,v in l.items() if len(v) > 0}

        for k in l.keys():
            l[k] = resample(l[k], 100).tolist()

        for k, div, prec in [('mem', 1e9, 2), ('gpu_mem', 1e9, 3), ('cpu', 1, 1), ('mem', 1, 1)]:
            if k in l:
                l[k] = [round(x / div, prec) for x in l[k]]

        del l['time']

        out = json.dumps(l)
        return out
    except BaseException:
        return json.dumps([])
    

@app.route('/metrics/<query>')
def metrics(query):
    
    try:

        m = json.load(open(join('logs/', query, 'metrics.json')))
        m_iter = sorted([int(k) for k in m.keys()])
        metric_names = list(next(iter(m.values())).keys())

        metric_vals = {name: [m[str(k)][name] for k in m_iter if name in m[str(k)]] for name in metric_names}

        # metric_vals = {name: resample(vals, 100) for name, vals in metric_vals.items()}
        metric_vals = {name: [np.round(v, 4) for v in vals] for name, vals in metric_vals.items()}

        return json.dumps(metric_vals)
    except (KeyError, FileNotFoundError, AttributeError) as e:
        print(e)
        return json.dumps([[],[]])

# if __name__ == "__main__":
#     app.run(debug=True, port=8000)
