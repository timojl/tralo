import inspect
from datetime import datetime
from tralo.utils import filter_args, sha1_hash_object, valid_run, AttributeDict, get_attribute
import yaml
import os
import json
import re
import torch
from os.path import join, isfile, expanduser, realpath
from tralo.log import log


def load_model(checkpoint_id, weights_file=None, strict=True, model_args='from_config', with_config=False):

    config = json.load(open(join('logs', checkpoint_id, 'config.json')))

    if model_args != 'from_config' and type(model_args) != dict:
        raise ValueError('model_args must either be "from_config" or a dictionary of values')

    model_cls = get_attribute(config['model'])

    # load model
    if model_args == 'from_config':
        _, model_args, _ = filter_args(config, inspect.signature(model_cls).parameters)

    print(model_args)

    model = model_cls(**model_args)

    if weights_file is None:
        weights_file = realpath(join('logs', checkpoint_id, 'weights.pth'))
    else:
        weights_file = realpath(join('logs', checkpoint_id, weights_file))

    if isfile(weights_file):
        weights = torch.load(weights_file)
        for _, w in weights.items():
            assert not torch.any(torch.isnan(w)), 'weights contain NaNs'
        model.load_state_dict(weights, strict=strict)
    else:
        raise FileNotFoundError(f'model checkpoint {weights_file} was not found')

    if with_config:
        return model, config
    
    return model


class Results(dict):
    """ results representation allowing html or print output """

    def _repr_html_(self):

        table, cols, diff_cols = self.table()

        # table to string, needed because of fixed column width
        table = [{k: self.to_str(v) for k, v in row.items()} for row in table]

        tab = '<table>'
        tab += '<thead>' + ''.join(f'<td>{k}</td>' for k in diff_cols) + '</thead>'
        tab += ''.join('<tr>' + ''.join(f'<td>{row[k]}</td>' for k in diff_cols) + '</tr>' for row in table)
        tab += '</table>'

        return tab

    def table(self):

        # scores_flat = [{f'{kk}': val for kk, val in s} for s in self['scores']]

        # import ipdb
        # ipdb.set_trace()

        def name(k, kk):
            return f'{k}_{kk}' if len(k) > 0 else kk

        scores_flat = [dict((name(k, kk), val) for k, sc in config_scores for kk, val in sc) 
                       for config_scores in self['scores']]

        # scores_flat = [{f'{k}_{kk}': val for k in s for kk, val in s[k]} for s in self['scores']]
        table = [{**a, **b, **c} for a, b,c  in zip(self['configurations'], scores_flat, self['stats'])]

        cols = list()
        [cols.append(k) if k not in cols else None for row in table for k in row.keys()]

        # re-order the columns
        first_col_order = ['name']
        cols = sorted(cols, key=lambda k: first_col_order.index(k) - 10000 if k in first_col_order else cols.index(k))
        print()

        # make sure all cols have values
        table = [{k: row[k] if k in row else None for k in cols} for row in table]

         # identify columns that have different values, use str to be compatible with list/tuples
        different_cols = [c for c in cols if len(set(str(tab[c]) if c in tab else None for tab in table)) > 1]

        return table, cols, different_cols

    def dataframe(self):
        from pandas import DataFrame

        table, _, _ = self.table()
        return DataFrame(table)

    def run_names(self):
        return [c['name'] for c in self['configurations']]

    def to_str(self, x):

        if type(x) == float:
            return f'{x:.5f}'

        if x is None:
            return '-'

        return str(x)

    def print(self, all=False, markdown=False):
        table, all_cols, diff_cols = self.table()

        if all:
            cols = all_cols
        elif self['visible_columns']:
            cols = self['visible_columns']
        else:
            cols = diff_cols

        col_spacer = ' ' if not markdown else ' | '

        # table to string, needed because of fixed column width
        table = [{k: self.to_str(v) for k, v in row.items()} for row in table]


        col_sizes = [max(len(tab[c]) if c in tab else 5 for tab in table) for c in cols]
        col_format = col_spacer.join("{:<" + str(max(s, len(col_name)) + 1) + "}" for s, col_name in zip(col_sizes, cols))

        all_columns = set(col_name for row in table for col_name in row.keys())
        log.hint('Available columns:', ', '.join(sorted(list(all_columns))))

        print()
        print(col_format.format(*cols))
        if markdown:
            # avoid confusion with itemize
            table = [{k: '--' if v == '-' else v for k, v in row.items()} for row in table] 
            print(' | '.join(['-'*len(c) for c in cols]))

        missing  = [(k, i) for i, row in enumerate(table) for k in cols if k not in row]            
        if len(missing) > 0:
            raise ValueError('The following columns (rows in brackets) are missing: ' + ', '.join(f'{k} ({i})' for k, i in missing))

        for row in table:
            print(col_format.format(*[row[k] for k in cols]))


def experiment(experiment_file, retrain=False, retest=False, nums=None, no_log=True, 
               verify=False, no_train=False):
   
    # add the current working dir as the first place to search for modules
    import sys
    sys.path = [os.getcwd()] + sys.path

    if no_log:
        import tralo
        tralo.log.log.level = 'warning'

    experiment = yaml.load(open(experiment_file), Loader=yaml.SafeLoader)
    configurations = experiment['individual_configurations']
    numbers = range(len(configurations))

    # number of individual configurations must match
    if 'individual_test_configurations' in experiment:
        assert len(experiment['individual_test_configurations']) == len(experiment['individual_configurations'])

    # if retrain => retest
    retest = retest or retrain

    # special experiment arguments
    if nums is not None:
        try:
            start, end = int(nums), int(nums) + 1
        except ValueError:
            assert type(nums) == str and ':' in nums
            start, end = nums.split(':')
            start, end = int(start) if len(start) > 0 else None, int(end) if len(end) > 0 else None
    else:
        start, end = 0, None

    numbers = numbers[start: end]
    configurations = configurations[start: end]

    checkpoint_ids = []
    for i, config in zip(numbers, configurations):

        # local config (config) overwrites global config (experiment['configuration'])
        config = AttributeDict({**experiment['configuration'], **config})

        train_fun = get_attribute(config['trainer'])
        score_fun = get_attribute(config['scorer'])
        del config['trainer'], config['scorer']

        # identifier for training
        checkpoint_id = sha1_hash_object(config)[:10] if 'name' not in config else config.name
        checkpoint_ids += [checkpoint_id]
        # train_configs += [config]

        os.makedirs(join('logs', checkpoint_id), exist_ok=True)

        # if currently training, retrain fails
        if isfile(join('logs', checkpoint_id, 'stats.json')):
            stats = json.load(open(join('logs', checkpoint_id, 'stats.json')))
            if 'start_time' in stats and 'stop_time' not in stats:
                log.info(f'experiment {i} {checkpoint_id}: currently training')
                raise SystemError(config['name'] + ': Cannot retrain while checkpoint is training.')

        # clean old scores
        if retrain:
            for filename in os.listdir(join('logs', checkpoint_id)):
                if re.match(r'^scores_[\w-]{10}\.json$', filename):
                    os.remove(join('logs', checkpoint_id, filename))

        # try to find the model
        if valid_run(join('logs', checkpoint_id)) and not retrain:
            log.info(f'experiment {i} {checkpoint_id}: already trained')
        elif config.max_iterations == 0:
            pass
        else:
            if not no_train:
                log.info(f'experiment {i} {checkpoint_id} start training')
                log.info('\n' + ''.join([f'{k}: {v}\n' for k,v in config.items()]))
                train_fun(config)

    assert len(configurations) == len(checkpoint_ids)

    log.info('#'*25 + '\nTraining complete\n')

    if verify:
        # shows differences between current experiment configuration
        # and the configuration that was used to train the checkpoint.
        for i, train_config, train_checkpoint_id in zip(numbers, configurations, checkpoint_ids):
            config_file = join('logs', train_checkpoint_id, f'config.json')
            config = json.load(open(config_file))

            train_config = AttributeDict({**experiment['configuration'], **train_config})

            # TODO: make sure that trainer and scorer can actually be ignored here
            all_keys = sorted(set(train_config.keys()).union(config.keys()).difference(['trainer', 'scorer']))
            log.info(f'experiment {i}: {train_config["name"]}')
            for k in all_keys:
                if k not in train_config:
                    log.info(f'  {k} missing in current config')
                elif k not in config:
                    log.info(f'  {k} missing in checkpoint')
                elif train_config[k] != config[k]:
                    log.info(f'  {k}: current: {config[k]}, checkpoint {train_config[k]}')

        return Results(scores=[], configurations=[])
            #print(config, train_config)


    # general test arguments, must be list of dicts
    test_arguments = experiment['test_configuration'] if 'test_configuration' in experiment else [dict()]
    test_arguments = [test_arguments] if type(test_arguments) == dict else test_arguments

    test_arguments_common = experiment['test_configuration_common'] if 'test_configuration_common' in experiment else dict()

    all_scores = []
    for i, train_config, train_checkpoint_id in zip(numbers, configurations, checkpoint_ids):

        # TODO: remove individual_test_configurations, it is easier to specify this directly in the configuration
        indiv_config = experiment['individual_test_configurations'][i] if 'individual_test_configurations' in experiment else [dict()]* len(test_arguments)
        indiv_config = [indiv_config] if type(indiv_config) == dict else indiv_config

        assert len(test_arguments) == len(indiv_config)

        this_train_config = AttributeDict({**experiment['configuration'], **train_config})

        # allow a test_configuration attribute in the training configuration
        if this_train_config.test_configuration is not None:
            indiv_config = [{**a, **this_train_config.test_configuration} for a in indiv_config]

        this_scores = []
        for j, (this_test_arguments, this_indiv_config) in enumerate(zip(test_arguments, indiv_config)):

            # overwrite order: common, per_test_dataset, per_model
            config = AttributeDict({**test_arguments_common, **this_test_arguments, **this_indiv_config})
            log.info('Test config:', config)
            checkpoint_id = sha1_hash_object(config)[:10]
            score_file = join('logs', train_checkpoint_id, f'scores_{checkpoint_id}.json')

            # evaluation
            if isfile(score_file) and not retest:
                score_file_date = datetime.fromtimestamp(os.stat(score_file).st_mtime).strftime("%d.%m.%y %H:%M")
                log.info(f'Found scores for configuration {i}.{j} (hash: {checkpoint_id}, date: {score_file_date})')
                scores = json.load(open(score_file))
            else:
                log.info(f'Compute scores for configuration {i}.{j} (hash: {checkpoint_id})')
                scores = score_fun(config, train_checkpoint_id, this_train_config)  # dictionary of scores
                json.dump(scores, open(score_file, 'w'))

            # handling of different output types of scores:
            # can be either dict or list/tuple and have different depths
            # scores per metric and dataset or per metric only.

            # scores are per metric only
            if all([type(s) in {list, tuple} and len(s) == 2 for s in scores]):
                scores = [['', scores]]

            # transforms dictionaries into lists of tuples
            if type(scores) == dict:
                scores = list(scores.items())

            scores = [(k, list(sc.items())) if type(sc) == dict else (k, sc) for k, sc in scores]

            # make sure scores are not overwritten if same name is used
            this_scores += [(k if k not in this_scores else f'{k}_{j}', v) for k,v in scores]
        all_scores += [this_scores]

    # load training stats and configs from disk
    all_stats, train_configs = [], []
    for i, train_checkpoint_id in zip(numbers, checkpoint_ids):

        try:
            losses_file = join('logs', train_checkpoint_id, f'losses.json')
            losses = json.load(open(losses_file))
            last10_loss = losses['loss'][-int(0.1*len(losses['loss'])):]
        except FileNotFoundError:
            last10_loss = None
            log.info(f'No losses.json found for {train_checkpoint_id}')

        # load training statistics
        run_stats = dict()
        try:
            stats_file = join('logs', train_checkpoint_id, f'stats.json')
            stats = json.load(open(stats_file))

            run_stats.update({
                'i': i,
                'learn_params': f'{stats["learn_params"]/1e6:.4f}M',
                'duration': f'{(stats["stop_time"] - stats["start_time"])/60:.1f}min',
                'date': datetime.fromtimestamp(stats['start_time']).strftime('%d.%m %H:%M'),
                'train_loss': torch.mean(torch.tensor(last10_loss)).item(),
            })
        except (FileNotFoundError, KeyError):
            log.info(f'No stats.json found for {train_checkpoint_id}')

        try:
            metrics = json.load(open(join('logs/', train_checkpoint_id, 'metrics.json')))
            
            all_metrics = list(set(x for v in metrics.values() for x in v.keys()))
            for k in all_metrics:
                k_new, reduce = ('loss', min) if k == 'val_loss' else (k, max)
                run_stats['val_'+ k_new] = reduce([v[k] for v in metrics.values()])

        except (FileNotFoundError, KeyError):
            log.info(f'No metrics.json found/extraction for {train_checkpoint_id}')

        all_stats += [run_stats]

        try:
            config_file = join('logs', train_checkpoint_id, f'config.json')
            config = json.load(open(config_file))
            train_configs += [config]
        except FileNotFoundError:
            train_configs += [{}]

    columns = experiment['columns'] if 'columns' in experiment else None
    return Results(scores=all_scores, stats=all_stats, configurations=train_configs, 
                   visible_columns=columns)

