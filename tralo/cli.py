import shutil
from os.path import realpath, dirname, join, isdir
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=['server', 'experiment', 'exp', 'create'])
    args, args_extra = parser.parse_known_args()

    if args.mode == 'server':
        from tralo.server import app
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=8001)
        args2 = parser.parse_args(args_extra)

        app.run(debug=True, port=args2.port)

    elif args.mode in {'experiment', 'exp'}:
        parser = argparse.ArgumentParser()
        parser.add_argument("experiment_file", type=str)
        parser.add_argument("--nums", type=str, default=None)
        parser.add_argument("--retrain", action='store_true')
        parser.add_argument("--retest", action='store_true')
        parser.add_argument("--no_log", action='store_true')
        parser.add_argument("--no_train", action='store_true')
        parser.add_argument("--markdown", action='store_true')
        parser.add_argument("--verify", default=False, action='store_true', help="check if current training arguments match with the actually used training arguments")

        args2 = parser.parse_args(args_extra)
        from tralo.experiments import experiment
        res = experiment(args2.experiment_file, nums=args2.nums, no_log=args2.no_log, verify=args2.verify, 
                         retrain=args2.retrain, retest=args2.retest, no_train=args2.no_train)
        
        if len(res['configurations']) > 1:
            res.print(markdown=args2.markdown)
        else:
            print(res['scores'])

    elif args.mode == 'create':
        from tralo.log import log
        import os
        template_dir = realpath(join(dirname(realpath(__file__)), '..', 'templates'))

        parser = argparse.ArgumentParser()
        parser.add_argument("template", type=str, help='Name of the template')
        parser.add_argument("name", type=str, help='Project name')
        args2 = parser.parse_args(args_extra)

        template = join(template_dir, args2.template)
        if isdir(template):
            shutil.copytree(template, join(os.getcwd(), args2.name))
            log.info(f'created project {args2.name}')
        else:
            raise FileNotFoundError(f'The directory {template} does not exist')
    else:
        raise ValueError(f'Invalid mode: {args.mode}')

if __name__ == '__main__':
    main()
