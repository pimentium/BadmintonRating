#!/usr/bin/python
import argparse
import json
import collections
import sys

import datetime
import numpy as np
import hyperopt
from hyperopt import hp
from scipy import stats

#
# Ideas:
# - Regularization
# - Product in double rating prediction: a_0 + a_1 * r_1 + a_2 * r_2 + a_3 * r_1^b_1 * r_2^b_2
# - Use score values
# - Aggregate by day
# - Adaptive learning rate
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
    def __init__(self, value, setter, partial_derivative, learning_rate):
        self.value = value
        self.setter = setter
        self.partial_derivative = partial_derivative
        self.learning_rate = learning_rate


def get_setter(collection, key):
    def setter(value):
        assert value == value
        collection[key] = value

    return setter


class Model(object):
    INITIAL_RATING = 1200
    SIGMOID_SCALE = np.log(10) / 400

    CHECK_GRADIENT = False

    def __init__(self, parameters):
        self.parameters = parameters
        self.single_ratings = collections.defaultdict(lambda: Model.INITIAL_RATING)
        self.double_ratings = collections.defaultdict(lambda: self.parameters.double_initial_rating)
        self.team_play_counts = collections.Counter()

    def predict(self, first_team, second_team, date):
        first_rating, _ = self.get_team_rating_and_updater(first_team)
        second_rating, _ = self.get_team_rating_and_updater(second_team)
        return sigmoid(Model.SIGMOID_SCALE * (first_rating - second_rating))

    def update(self, record):
        first_rating, first_updater = self.get_team_rating_and_updater(record.first_team)
        second_rating, second_updater = self.get_team_rating_and_updater(record.second_team)
        delta = first_rating - second_rating
        sign = 1 if record.first_score > record.second_score else -1
        derivative = sigmoid(-sign * Model.SIGMOID_SCALE * delta)
        first_updater(sign * Model.SIGMOID_SCALE * derivative)
        second_updater(-sign * Model.SIGMOID_SCALE * derivative)

    def get_team_rating_and_updater(self, team):
        rating, variables = self.get_team_rating_and_variables(team)
        if Model.CHECK_GRADIENT:
            self.check_gradient(team, variables)

        def update(derivative):
            for variable in variables:
                variable.setter(variable.value + variable.learning_rate * variable.partial_derivative * derivative)
            self.team_play_counts[tuple(sorted(team))] += 1

        return rating, update

    def get_team_rating_and_variables(self, team):
        if len(team) == 1:
            player = team[0]
            rating = self.single_ratings[player]
            return rating, [Variable(rating, get_setter(self.single_ratings, player), 1, self.parameters.learning_rate)]
        else:
            team = tuple(sorted(team))
            proper_rating = self.double_ratings[team]
            player1, player2 = team
            player1_rating = self.single_ratings[player1]
            player2_rating = self.single_ratings[player2]

            rating_diff = player1_rating - player2_rating
            scaled_rating_diff = self.parameters.single_weights_sigmoid_scale * rating_diff
            sig = sigmoid(scaled_rating_diff)
            player1_weight = ((1 - sig) * self.parameters.lower_single_weight +
                              sig * self.parameters.higher_single_weight)
            player2_weight = (sig * self.parameters.lower_single_weight +
                              (1 - sig) * self.parameters.higher_single_weight)

            sig_derivative = sigmoid_derivative(scaled_rating_diff)
            weight_diff = self.parameters.higher_single_weight - self.parameters.lower_single_weight

            team_count = self.team_play_counts[team]
            weight_sum = self.parameters.single_weight + team_count
            rating = (self.parameters.single_weight * (player1_weight * player1_rating +
                                                       player2_weight * player2_rating) +
                      team_count * proper_rating) / weight_sum

            return rating, [
                Variable(proper_rating, get_setter(self.double_ratings, team),
                         team_count / weight_sum,
                         self.parameters.double_learning_rate),
                Variable(player1_rating, get_setter(self.single_ratings, player1),
                         self.parameters.single_weight / weight_sum * (
                             player1_weight + weight_diff * sig_derivative * scaled_rating_diff),
                         self.parameters.double_single_learning_rate),
                Variable(player2_rating, get_setter(self.single_ratings, player2),
                         self.parameters.single_weight / weight_sum * (
                             player2_weight - weight_diff * sig_derivative * scaled_rating_diff),
                         self.parameters.double_single_learning_rate)
            ]

    def check_gradient(self, team, variables):
        for variable in variables:
            old_value = variable.value
            variable.setter(old_value - 1e-7)
            low_rating, _ = self.get_team_rating_and_variables(team)
            variable.setter(old_value + 1e-7)
            high_rating, _ = self.get_team_rating_and_variables(team)
            variable.setter(old_value)
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
    def __init__(self, label, default_value, hp_variable, log_pdf):
        self.label = label
        self.default_value = default_value
        self.hp_variable = hp_variable
        self.log_pdf = log_pdf

    @staticmethod
    def normal_from_bounds(label, left_bound, right_bound, quantization=None):
        mean = (left_bound + right_bound) / 2.0
        sigma = (right_bound - left_bound) / 4.0
        hp_variable = (hp.normal(label, mean, sigma) if quantization is None
                       else hp.qnormal(label, mean, sigma, quantization))
        return Parameter(label, mean, hp_variable, stats.norm(mean, sigma).logpdf)

    @staticmethod
    def log_normal_from_bounds(label, left_bound, right_bound, quantization=None):
        log_left_bound = np.log(left_bound)
        log_right_bound = np.log(right_bound)
        log_mean = (log_left_bound + log_right_bound) / 2.0
        log_sigma = (log_right_bound - log_left_bound) / 4.0
        mean = np.exp(log_mean)
        hp_variable = (hp.lognormal(label, log_mean, log_sigma) if quantization is None
                       else hp.qlognormal(label, log_mean, log_sigma, quantization))
        return Parameter(label, mean, hp_variable, stats.lognorm(log_sigma, scale=mean).logpdf)


class Parameters(object):
    PARAMETERS = [
        Parameter.normal_from_bounds('learning_rate', 10 / Model.SIGMOID_SCALE, 100 / Model.SIGMOID_SCALE),
        Parameter.normal_from_bounds('double_initial_rating', 1250, 2300),
        Parameter.log_normal_from_bounds('double_learning_rate', 10 / Model.SIGMOID_SCALE, 100 / Model.SIGMOID_SCALE),
        Parameter.log_normal_from_bounds('single_weight', 1, 30),
        Parameter.log_normal_from_bounds('higher_single_weight', 0.5, 0.95),
        Parameter.log_normal_from_bounds('lower_single_weight', 0.3, 0.9),
        Parameter.log_normal_from_bounds('single_weights_sigmoid_scale', 0.01, 1),
        Parameter.log_normal_from_bounds('double_single_learning_rate',
                                         10 / Model.SIGMOID_SCALE, 100 / Model.SIGMOID_SCALE)
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


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x)) if x >= 0 else 1.0 - 1.0 / (1 + np.exp(x))


def sigmoid_derivative(x):
    exp = np.exp(-abs(x))
    return exp / (1 + exp) ** 2


def evaluate_model(model, records, valid_range=None):
    count = 0
    ll_sum = 0.0
    correct_count = 0
    capital = 1.0
    for index, record in enumerate(records):
        if valid_range is None or valid_range(index):
            p = model.predict(record.first_team, record.second_team, record.date)
            assert 0 <= p <= 1
            assert record.first_score != record.second_score
            p_corrected = p if record.first_score > record.second_score else 1 - p
            if p_corrected > 0.5:
                correct_count += 1
            ll_sum += np.log(p_corrected)
            bet = capital
            capital += bet * (p_corrected / ((p_corrected * bet + 0.5) / (bet + 1)) - 1)
            count += 1
        model.update(record)
    return {
        'Count': count,
        'LogLikelihood': ll_sum / count,
        'Likelihood': np.exp(ll_sum / count),
        'Precision': float(correct_count) / count,
        'Capital': capital
    }


def evaluate_parameters(parameters, records, valid_range=None):
    model = Model(parameters)
    result = evaluate_model(model, records, valid_range)
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
    for record in records:
        model.update(record)
    model.print_info(args.output)
    if args.output is not sys.stdout:
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
    for key, value in best_parameters.iteritems():
        print '%s: %s' % (key, value)
    json.dump(best_parameters, args.output)
    args.output.write('\n')


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


def cross_validate(args):
    records = list(read(args.input))
    average_result = None
    whole_range = range(len(records))
    if args.range is not None:
        whole_range = [i for i in whole_range if args.range(i)]
    fold_count = args.folds
    fold_range_len = len(whole_range) / fold_count
    for fold in xrange(fold_count):
        fold_range = whole_range[fold * fold_range_len:(fold + 1) * fold_range_len]
        print '.',
        sys.stdout.flush()
        # print 'Fold %d: [%d, %d]' % (fold, fold_range[0], fold_range[-1])
        parameters, _ = tune_parameters(
            records,
            lambda j: j in whole_range and j not in fold_range,
            args.regularizer, args.tunetarget, args.max_evals, args.seed)
        result = evaluate_parameters(Parameters.from_dict(parameters), records, lambda j: j in fold_range)
        # for key, value in result.iteritems():
        #     print '%s: %s' % (key, value)
        # print
        if average_result is None:
            average_result = {key: 0 for key in result.iterkeys()}
        for key in result.iterkeys():
            average_result[key] += float(result[key]) / fold_count
    print
    # print 'Average results'
    for key, value in average_result.iteritems():
        print '%s: %s' % (key, value)


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
    parser.add_argument('--output', type=argparse.FileType('w'), default=sys.stdout,
                        help='File to write ratings or parameters')
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
    parser.add_argument('--max_evals', type=int, default=100,
                        help='Maximal number of function evaluations during tuning')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed')
    parser.set_defaults(tunetarget='Capital')
    parser.add_argument('--ll', dest='tunetarget', action='store_const', const='LogLikelihood',
                        help='Tune log-likelihood instead of capital')
    parser.add_argument('--reg', dest='regularizer', type=float, default=0.0,
                        help='Regularizer used in tuning')
    parser.add_argument('--folds', type=int, default=10,
                        help='Number of folds for cross-validation')
    args = parser.parse_args()
    np.random.seed(args.seed)
    if args.checkgrad:
        Model.CHECK_GRADIENT = True
    args.func(args)


if __name__ == '__main__':
    main()
