#!/usr/bin/env python
# coding=utf-8
""" task_evaluate.py - Scoring program for Fincausal 2025 Task

    (Adapted from Fincausal 2023 to deal with the background set)

    usage: task_evaluate.py [-h] {from-folder,from-file} ...

    positional arguments:
      {from-folder,from-file}
                            Use from-file for basic mode or from-folder for
                            Codalab compatible mode

Usage 1: Folder mode

    usage: task_evaluate.py from-folder [-h] input output

    Codalab mode with input and output folders

    positional arguments:
      input       input folder with ref (reference) and res (result) sub folders
      output      output folder where score.txt is written

    optional arguments:
      -h, --help  show this help message and exit
    task_evaluate input output

    input, output folders must follow the Codalab competition convention for scoring bundle
    e.g.
        ├───input
        │   ├───ref
        │   └───res
        └───output

Usage 2: File mode

    usage: task_evaluate.py from-file [-h] ref_file pred_file [score_file]

    Basic mode with path to input and output files

    positional arguments:
      ref_file    reference file
      pred_file   prediction file to evaluate
      score_file  path to output score file (or stdout if not provided)

    optional arguments:
      -h, --help  show this help message and exit
"""

import os
import sys
import csv
import argparse
import logging
import numpy as np
# import pip

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package], stdout=sys.stderr)

#install("sentence_transformers")


from sentence_transformers import SentenceTransformer, util
from collections import namedtuple

model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
st_model = SentenceTransformer(model)

Task2Data = namedtuple('Task2Data', ['index', 'context', 'question', 'answer'])

def SAS(predicted_answers, reference_answers):
    predictions_embeddings = st_model.encode(predicted_answers, convert_to_tensor=True)
    reference_embeddings = st_model.encode(reference_answers, convert_to_tensor=True)
    similarity_scores = [util.cos_sim(p, l).cpu().numpy() for p, l in zip(predictions_embeddings, reference_embeddings)]
    return np.mean(similarity_scores)

def ExactMatch(predicted_answers, reference_answers):
    return np.mean([1 if pred.lower().strip() == ref.lower().strip() else 0 for pred, ref in zip(predicted_answers, reference_answers)])

def evaluate(truth, predict, classes):
    """
    Fincausal 2022 Task evaluation: returns precision, recall and F1 comparing submitting data to reference data.
    :param truth: list of Task2Data(index, context, question, answer) - reference data set
    :param predict: list of Task2Data(index, context, question, answer) - submission data set
    :param classes: list of classes
    :return: SAS, ExactMatch
    """
    pred_answers = [pred.answer for pred in predict]
    ref_answers = [ref.answer for ref in truth]
    return SAS(ref_answers, pred_answers), ExactMatch(ref_answers, pred_answers) 

def get_data(csv_lines):
    """
    Retrieve Task 2 data from CSV content (separator is ';') as a list of (index, text, cause, effect).
    :param csv_lines:
    :return: list of Task2Data(index, text, cause, effect, labels)
    """
    result = []

    for line in csv_lines:
        line = line.rstrip('\n')

        r = csv.reader([line], delimiter=";", quotechar='"')
        r = list(r)

        [index, context, question, answer] = r[0]

        context = context.lstrip()
        question = question.lstrip()
        answer = answer.lstrip()

        result.append(Task2Data(index, context, question, answer))

    return result


def evaluate_files(gold_file, submission_file, output_file=None):
    """
    Evaluate Precision, Recall, F1 scores between gold_file and submission_file
    If output_file is provided, scores are saved in this file and printed to std output.
    :param gold_file: path to reference data
    :param submission_file: path to submitted data
    :param output_file: path to output file as expected by Codalab competition framework
    :return:
    """
    if os.path.exists(gold_file) and os.path.exists(submission_file):
        logging.info("Gold file: " + gold_file)
        with open(gold_file, 'r', encoding='utf-8') as fp:
            ref_csv = []
            try:
                for line in fp:
                    line = line.strip()
                    logging.info("Gold: " + line[:30])
                    ref_csv.append(line)
            except UnicodeDecoderError as e:
                logging.info(e)

        logging.info("Submission file: " + submission_file)
        with open(submission_file, 'r', encoding='utf-8') as fp:
            sub_csv = []
            try:
                for line in fp:
                    line = line.strip()
                    logging.info("Sub: " + line[:30])
                    sub_csv.append(line)
            except UnicodeDecoderError as e:
                logging.info(e)

        # JPZ: get test ids
        valid_ids = {row.split(";")[0] for row in ref_csv}
        
        # JPZ: remove the background set rows
        sub_csv = [row for row in sub_csv if row.split(";")[0] in valid_ids]

        #print(ref_csv)
        #print(sub_csv)
        
        # Get data (no skipping headers)
        logging.info('* Loading reference data')
        y_true = get_data(ref_csv[0:]) 

        logging.info('* Loading prediction data')
        y_pred = get_data(sub_csv[0:])

        logging.info(f'Load Data: check data set length = {len(y_true) == len(y_pred)}')
        logging.info(f'Load Data: check data set ref. context = {all([x.context == y.context for x, y in zip(y_true, y_pred)])}')
        assert len(y_true) == len(y_pred), f"{len(y_true)} / {len(y_pred)}"
        assert all([x.context == y.context for x, y in zip(y_true, y_pred)])
        
        # Process data using classes: -, C & E

        SAS, exact_match = evaluate(y_true, y_pred, ['-', 'C', 'E'])

        scores = [
            "SAS: %f\n" % SAS,
            "ExactMatch: %f\n" % exact_match
        ]

        for s in scores:
            print(s, end='')
        
        if output_file is not None:
            with open(output_file, 'w', encoding='utf-8') as fp:
                for s in scores:
                    fp.write(s)
    else:
        # Submission file most likely being the wrong one - tell which one we are looking for
        logging.error(f'{os.path.basename(gold_file)} not found')


def from_folder(args):
    # Folder mode - Codalab usage
    submit_dir = os.path.join(args.input, 'res')
    truth_dir = os.path.join(args.input, 'ref')
    output_dir = args.output

    if not os.path.isdir(submit_dir):
        logging.error("%s doesn't exist" % submit_dir)

    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    o_file = os.path.join(output_dir, 'scores.txt')

    gold_list = os.listdir(truth_dir)
    for gold in gold_list:
        g_file = os.path.join(truth_dir, gold)
        s_file = os.path.join(submit_dir, gold)
        evaluate_files(g_file, s_file, o_file)

    return 0


def from_file(args):
    return evaluate_files(args.ref_file, args.pred_file, args.score_file)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Use from-file for basic mode or from-folder for Codalab compatible mode')

    command1_parser = subparsers.add_parser('from-folder', description='Codalab mode with input and output folders')
    command1_parser.set_defaults(func=from_folder)
    command1_parser.add_argument('input', help='input folder with ref (reference) and res (result) sub folders')
    command1_parser.add_argument('output', help='output folder where score.txt is written')

    command2_parser = subparsers.add_parser('from-file', description='Basic mode with path to input and output files')
    command2_parser.set_defaults(func=from_file)
    command2_parser.add_argument('ref_file', help='reference file')
    command2_parser.add_argument('pred_file', help='prediction file to evaluate')
    command2_parser.add_argument('score_file', nargs='?', default=None,
                                 help='path to output score file (or stdout if not provided)')

    logging.basicConfig(level=logging.INFO,
                        filename=None,
                        format='%(levelname)-7s| %(message)s')

    args = parser.parse_args()
    if 'func' in args:
        args.func(args)
        exit(0)
    else:
        parser.print_usage()
        exit(1)    

if __name__ == '__main__':
    main()

