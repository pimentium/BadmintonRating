#!/usr/bin/python
import argparse
import json
import collections
import multiprocessing
import random
import sys

import datetime
import itertools
import numpy as np
import hyperopt
from hyperopt import hp
from scipy import stats


#
# Ideas:
# - Bet_p optimization
# - Playing with parameters
# - Use score values
# - Aggregate by day
# - Bet prediction
#


class Record(object):
    def __init__(self, first_team, second_team, date, first_score, second_score):
        self.first_team = first_team
        self.second_team = second_team
        self.date = date
        self.first_score = first_score
        self.second_score = second_score


class Variable(object):
    def __init__(self, updater, partial_derivative, learning_rate):
        self.updater = updater
        self.partial_derivative = partial_derivative
        self.learning_rate = learning_rate

    def update(self, delta):
        self.updater(self.learning_rate * self.partial_derivative * delta)


def get_updater(collection, key):
    def updater(value):
        collection[key] += value

    return updater


class Model(object):
    INITIAL_RATING = 1200
    SIGMOID_SCALE = np.log(10) / 400

    CHECK_GRADIENT = False

    def __init__(self, parameters):
        self.parameters = parameters
        self.single_ratings = collections.defaultdict(lambda: Model.INITIAL_RATING)
        self.double_ratings = collections.defaultdict(lambda: self.parameters.double_initial_rating)
        self.team_play_counts = collections.Counter()

    def predict_and_update(self, first_team, second_team, date):
        first_rating, first_updater = self.get_team_rating_and_updater(first_team)
        second_rating, second_updater = self.get_team_rating_and_updater(second_team)

        def update(first_score, second_score):
            delta = first_rating - second_rating
            sign = 1 if first_score > second_score else -1
            derivative = sigmoid(-sign * Model.SIGMOID_SCALE * delta)
            first_updater(sign * Model.SIGMOID_SCALE * derivative)
            second_updater(-sign * Model.SIGMOID_SCALE * derivative)

        return sigmoid(Model.SIGMOID_SCALE * (first_rating - second_rating)), update

    def get_team_rating_and_updater(self, team):
        rating, variables = self.get_team_rating_and_variables(team)
        if Model.CHECK_GRADIENT:
            self.check_gradient(team, variables)

        def update(derivative):
            for variable in variables:
                variable.update(derivative)
            self.team_play_counts[tuple(sorted(team))] += 1

        return rating, update

    def get_team_rating_and_variables(self, team):
        parameters = self.parameters
        if len(team) == 1:
            player = team[0]
            rating = self.single_ratings[player]
            return rating, [Variable(get_updater(self.single_ratings, player), 1, parameters.learning_rate)]
        else:
            team = tuple(sorted(team))
            proper_rating = self.double_ratings[team]
            player1, player2 = team
            assert player1 != player2
            player1_rating = self.single_ratings[player1]
            player2_rating = self.single_ratings[player2]

            player1_weight = parameters.single_weight
            player2_weight = parameters.single_weight

            team_count = self.team_play_counts[team]
            weight_sum = parameters.mixture_weight + team_count
            rating = (parameters.mixture_weight * (player1_weight * player1_rating + player2_weight * player2_rating +
                                                   parameters.double_rating_shift) +
                      team_count * proper_rating) / weight_sum

            return rating, [
                Variable(get_updater(self.double_ratings, team),
                         team_count / weight_sum,
                         parameters.double_learning_rate),
                Variable(get_updater(self.single_ratings, player1),
                         parameters.mixture_weight / weight_sum * player1_weight,
                         parameters.learning_rate),
                Variable(get_updater(self.single_ratings, player2),
                         parameters.mixture_weight / weight_sum * player2_weight,
                         parameters.learning_rate)
            ]

    def check_gradient(self, team, variables):
        for variable in variables:
            variable.updater(-1e-7)
            low_rating, _ = self.get_team_rating_and_variables(team)
            variable.updater(+2e-7)
            high_rating, _ = self.get_team_rating_and_variables(team)
            variable.updater(-1e-7)
            if abs((high_rating - low_rating) / 2e-7 - variable.partial_derivative) > 1e-5:
                self.get_team_rating_and_variables(team)
            assert abs((high_rating - low_rating) / 2e-7 - variable.partial_derivative) < 1e-5

    def print_info(self, output):
        print >> output, 'Ratings:'
        for key, value in sorted(self.single_ratings.iteritems(), key=lambda (k, v): v, reverse=True):
            print >> output, '%s: %.0f' % (key, value)
        print >> output, 'Pairs Ratings:'
        for (player1, player2), value in sorted(self.double_ratings.iteritems(), key=lambda (k, v): v, reverse=True):
            if self.single_ratings.get(player1) < self.single_ratings.get(player2):
                player1, player2 = player2, player1
            print >> output, '%s, %s: %.0f' % (player1, player2, value)


class Parameter(object):
    def __init__(self, label, default_value, hp_variable, log_pdf, cdf):
        self.label = label
        self.default_value = default_value
        self.hp_variable = hp_variable
        self.log_pdf = log_pdf
        self.cdf = cdf

    @staticmethod
    def normal_from_bounds(label, left_bound, right_bound, quantization=None):
        mean = (left_bound + right_bound) / 2.0
        sigma = (right_bound - left_bound) / 4.0
        hp_variable = (hp.normal(label, mean, sigma) if quantization is None
                       else hp.qnormal(label, mean, sigma, quantization))
        dist = stats.norm(mean, sigma)
        return Parameter(label, mean, hp_variable, dist.logpdf, dist.cdf)

    @staticmethod
    def log_normal_from_bounds(label, left_bound, right_bound, quantization=None):
        log_left_bound = np.log(left_bound)
        log_right_bound = np.log(right_bound)
        log_mean = (log_left_bound + log_right_bound) / 2.0
        log_sigma = (log_right_bound - log_left_bound) / 4.0
        mean = np.exp(log_mean)
        hp_variable = (hp.lognormal(label, log_mean, log_sigma) if quantization is None
                       else hp.qlognormal(label, log_mean, log_sigma, quantization))
        dist = stats.lognorm(log_sigma, scale=mean)
        return Parameter(label, mean, hp_variable, dist.logpdf, dist.cdf)


class Parameters(object):
    PARAMETERS = [
        Parameter.log_normal_from_bounds('learning_rate', 10 / Model.SIGMOID_SCALE, 100 / Model.SIGMOID_SCALE),
        Parameter.normal_from_bounds('double_initial_rating', 1250, 2300),
        Parameter.log_normal_from_bounds('double_learning_rate', 10 / Model.SIGMOID_SCALE, 100 / Model.SIGMOID_SCALE),
        Parameter.log_normal_from_bounds('mixture_weight', 1, 30),
        Parameter.log_normal_from_bounds('single_weight', 0.3, 0.7),
        Parameter.normal_from_bounds('double_rating_shift', -500, 100)
    ]

    def __init__(self, **kwargs):
        for parameter in Parameters.PARAMETERS:
            setattr(self, parameter.label, kwargs.get(parameter.label, parameter.default_value))

    @staticmethod
    def get_space():
        return {parameter.label: parameter.hp_variable for parameter in Parameters.PARAMETERS}

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(d):
        return Parameters(**d)

    def log_pdf(self):
        return sum(parameter.log_pdf(getattr(self, parameter.label)) for parameter in Parameters.PARAMETERS)

    def print_info(self):
        for parameter in Parameters.PARAMETERS:
            value = getattr(self, parameter.label)
            print '%s: %f (%f quantile)' % (parameter.label, value, parameter.cdf(value))

    @staticmethod
    def print_histograms(parameters):
        left_percentile, right_percentile = 5, 95
        for parameter in Parameters.PARAMETERS:
            values = [getattr(p, parameter.label) for p in parameters]
            values = np.sort(values)
            print '%s: [%f, %f]' % (parameter.label,
                                    values[left_percentile * len(values) / 100],
                                    values[right_percentile * len(values) / 100])


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x)) if x >= 0 else 1.0 - 1.0 / (1 + np.exp(x))


def sigmoid_derivative(x):
    exp = np.exp(-abs(x))
    return exp / (1 + exp) ** 2


def evaluate_model(model, records, valid_range=None):
    probabilities = []
    for index, record in enumerate(records):
        p, update = model.predict_and_update(record.first_team, record.second_team, record.date)
        if valid_range is None or valid_range(index):
            assert 0 <= p <= 1
            assert record.first_score != record.second_score
            probabilities.append(p if record.first_score > record.second_score else 1 - p)
        update(record.first_score, record.second_score)
    return probabilities


def calc_metrics(probabilities):
    probabilities = np.array(probabilities, copy=False)
    count = len(probabilities)
    log_likelihood = np.log(probabilities).mean()
    capital = 1.0
    for p in probabilities:
        bet = capital
        capital += bet * (p / ((p * bet + 0.5) / (bet + 1)) - 1)
    return {
        'Count': count,
        'LogLikelihood': log_likelihood,
        'Likelihood': np.exp(log_likelihood),
        'Precision': float(np.sum(probabilities > 0.5)) / count,
        'Capital': capital
    }


def evaluate_parameters(parameters, records, valid_range=None):
    model = Model(parameters)
    probabilities = evaluate_model(model, records, valid_range)
    result = calc_metrics(probabilities)
    result['ParametersLogPDF'] = parameters.log_pdf()
    return result


def read(filename):
    with open(filename) as data_file:
        data_file.readline()
        for line in data_file:
            tokens = line.split(',')
            date = datetime.datetime.strptime(tokens[0], '%m/%d/%Y %H:%M:%S')
            first_team = (tokens[1], tokens[2]) if tokens[2] else (tokens[1],)
            second_team = (tokens[3], tokens[4]) if tokens[4] else (tokens[3],)
            first_score = int(tokens[5])
            second_score = int(tokens[6])
            yield Record(first_team, second_team, date, first_score, second_score)


def process(args):
    model = Model(args.parameters)
    records = read(args.input)
    evaluate_model(model, records, lambda i: False)
    output = open(args.output, 'w') if args.output is not None else sys.stdout
    model.print_info(output)
    if args.output is not None:
        print 'Ratings are saved in', args.output


def evaluate(args):
    result = evaluate_parameters(args.parameters, read(args.input), args.range)
    for key, value in result.iteritems():
        print '%s: %s' % (key, value)


def tune(args):
    records = list(read(args.input))
    best_parameters, best_result = tune_parameters(records, args.range, args.regularizer, args.tunetarget,
                                                   args.max_evals, args.seed)
    print 'Best results'
    for key, value in best_result.iteritems():
        print '%s: %s' % (key, value)
    print 'Parameters'
    Parameters.from_dict(best_parameters).print_info()
    # for key, value in best_parameters.iteritems():
    #     print '%s: %s' % (key, value)
    if args.output is not None:
        with open(args.output, 'w') as output:
            json.dump(best_parameters, output)
            output.write('\n')


def tune_parameters(records, valid_range, regularizer, tune_target, max_evals, random_seed):
    def func(func_args):
        parameters = Parameters.from_dict(func_args)
        result = evaluate_parameters(parameters, records, valid_range)
        result['loss'] = -result[tune_target] - regularizer * result['ParametersLogPDF']
        result['status'] = hyperopt.STATUS_OK
        return result

    trials = hyperopt.Trials()
    hyperopt.fmin(func, Parameters.get_space(), algo=hyperopt.tpe.suggest, max_evals=max_evals, trials=trials,
                  rseed=random_seed)
    best_trial = trials.best_trial
    best_parameters = {key: value[0] for key, value in best_trial['misc']['vals'].iteritems()}
    best_result = best_trial['result']
    return best_parameters, best_result


def evaluate_fold(params):
    fold, records, whole_range, fold_range_len, regularizer, tunetarget, max_evals, seed = params
    fold_range = set(whole_range[fold * fold_range_len:(fold + 1) * fold_range_len])
    tune_range = set(whole_range) - set(fold_range)
    params_dict, _ = tune_parameters(
            records,
            lambda j: j in tune_range,
            regularizer, tunetarget, max_evals, seed)
    params = Parameters.from_dict(params_dict)
    model = Model(params)
    probabilities = evaluate_model(model, records, lambda j: j in fold_range)
    sys.stdout.write('.')
    sys.stdout.flush()
    return zip(sorted(fold_range), probabilities), params


def cross_validate(args):
    records = list(read(args.input))
    whole_range = range(len(records))
    random.shuffle(whole_range)
    if args.range is not None:
        whole_range = [i for i in whole_range if args.range(i)]
    fold_count = args.folds
    fold_range_len = (len(whole_range) + fold_count - 1) / fold_count

    params = [(fold, records, whole_range, fold_range_len, args.regularizer, args.tunetarget, args.max_evals, args.seed)
              for fold in xrange(fold_count)]
    if args.threads is None:
        results, parameters = zip(*map(evaluate_fold, params))
    else:
        pool = multiprocessing.Pool(args.threads)
        results, parameters = zip(*pool.map(evaluate_fold, params))
    print

    all_results = sorted(itertools.chain.from_iterable(results))
    if args.output is not None:
        with open(args.output, 'w') as output:
            for pair in all_results:
                print >> output, '%d\t%f' % pair

    probabilities = np.array(all_results, copy=False)[:, 1]
    count = len(probabilities)

    compare_probabilities = None
    if args.compare_to is not None:
        compare_probabilities = []
        for line in args.compare_to:
            index, prob = line.rstrip().split()
            index = int(index)
            prob = float(prob)
            compare_probabilities.append((index, prob))
        compare_probabilities.sort()
        assert [i for i, p in compare_probabilities] == [i for i, p in all_results], 'Different ranges'
        compare_probabilities = np.array(compare_probabilities, copy=False)[:, 1]

    bootstrap_count = 1000
    bootstrapped_results = collections.defaultdict(lambda: [])
    for i in xrange(bootstrap_count):
        indexes = np.random.choice(count, count, replace=True)
        metrics = calc_metrics(probabilities[indexes])
        if compare_probabilities is not None:
            compare_metrics = calc_metrics(compare_probabilities[indexes])
            for key in metrics.iterkeys():
                metrics[key] -= compare_metrics[key]
        for key, value in metrics.iteritems():
            bootstrapped_results[key].append(value)

    left_percentile, right_percentile = 5, 95
    print '[%d%%, %d%%] confidence intervals:' % (left_percentile, right_percentile)
    for key, values in bootstrapped_results.iteritems():
        values.sort()
        print '%s: [%f, %f]' % (key,
                                values[left_percentile * len(values) / 100],
                                values[right_percentile * len(values) / 100])

    print 'Parameters intervals:'
    Parameters.print_histograms(parameters)


def parse_range(s):
    start, end = s.split(':')
    if start and end:
        start = int(start)
        end = int(start)
        return lambda i: start <= i < end
    if start:
        start = int(start)
        return lambda i: start <= i
    if end:
        end = int(end)
        return lambda i: i < end
    return None


def read_parameters(filename):
    with open(filename) as parameters_file:
        params = json.load(parameters_file)
        return Parameters.from_dict(params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='records.csv',
                        help='File with input data')
    parser.add_argument('--output', '-o',
                        help='File to write ratings, parameters, or probabilities')
    parser.set_defaults(func=process)
    parser.add_argument('--evaluate', dest='func', action='store_const', const=evaluate,
                        help='Evaluate model')
    parser.add_argument('--tune', dest='func', action='store_const', const=tune,
                        help='Tune model parameters')
    parser.add_argument('--cv', dest='func', action='store_const', const=cross_validate,
                        help='Run cross validation on tuning')
    parser.add_argument('--range', type=parse_range,
                        help='Range of input records to evaluate, <start>:<end>')
    parser.add_argument('--params', dest='parameters', type=read_parameters, default=Parameters(),
                        help='File with model parameters')
    parser.add_argument('--checkgrad', default=False, action='store_true',
                        help='Check that gradient is correct on every update of the model')
    parser.add_argument('--max-evals', type=int, default=100,
                        help='Maximal number of function evaluations during tuning')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed')
    parser.set_defaults(tunetarget='LogLikelihood')
    parser.add_argument('--capital', dest='tunetarget', action='store_const', const='Capital',
                        help='Tune capital instead of log-likelihood')
    parser.add_argument('--reg', dest='regularizer', type=float, default=0.0,
                        help='Regularizer used in tuning')
    parser.add_argument('--folds', type=int, default=20,
                        help='Number of folds for cross-validation')
    parser.add_argument('--threads', '-j', type=int,
                        help='Number of threads for cross-validation')
    parser.add_argument('--compare-to', type=argparse.FileType(),
                        help='File with probabilities to compare current model with it')
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.checkgrad:
        Model.CHECK_GRADIENT = True
    args.func(args)


if __name__ == '__main__':
    main()
