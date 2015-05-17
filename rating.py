#!/usr/bin/python
import argparse
import json
import collections
import sys

import datetime
import numpy as np
from scipy import optimize


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
        if value != value:
            print 'A'
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
        first_rating = self.single_ratings.setdefault(first_team[0], Model.INITIAL_RATING)
        second_rating = self.single_ratings.setdefault(second_team[0], Model.INITIAL_RATING)
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
        # if len(team) == 1:
        #     rating = self.single_ratings[team[0]]
        #
        #     def update(dx):
        #         self.single_ratings[team[0]] = rating + self.parameters.learning_rate * dx
        # else:
        #     team = tuple(sorted(team))
        #     proper_rating = self.double_ratings[team]
        #     player1, player2 = team
        #     player1_rating = self.single_ratings[player1]
        #     player2_rating = self.single_ratings[player2]
        #
        #     if player1_rating > player2_rating:
        #         player1_weight = self.parameters.higher_single_weight
        #         player2_weight = self.parameters.lower_single_weight
        #     elif player1_rating < player2_rating:
        #         player1_weight = self.parameters.lower_single_weight
        #         player2_weight = self.parameters.higher_single_weight
        #     else:
        #         player1_weight = player2_weight = self.parameters.equal_single_weight
        #         # player1_weight = player2_weight = (self.parameters.lower_single_weight + self.parameters.higher_single_weight) / 2
        #
        #     team_count = self.double_counts[team]
        #     weight_sum = self.parameters.single_weight + team_count
        #     rating = (self.parameters.single_weight * (player1_weight * player1_rating +
        #                                                player2_weight * player2_rating) +
        #               team_count * proper_rating) / weight_sum
        #
        #     def update(dx):
        #         self.double_ratings[team] = (proper_rating + self.parameters.double_learning_rate *
        #                                      team_count / weight_sum * dx)
        #         self.single_ratings[player1] = (player1_rating + self.parameters.double_single_learning_rate *
        #                                         self.parameters.single_weight * player1_weight / weight_sum * dx)
        #         self.single_ratings[player2] = (player2_rating + self.parameters.double_single_learning_rate *
        #                                         self.parameters.single_weight * player2_weight / weight_sum * dx)
        #         self.double_counts[team] += 1

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
            player1_weight = (1 - sig) * self.parameters.lower_single_weight + sig * self.parameters.higher_single_weight
            player2_weight = sig * self.parameters.lower_single_weight + (1 - sig) * self.parameters.higher_single_weight

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
                         self.parameters.single_weight / weight_sum * (player1_weight + weight_diff * sig_derivative * scaled_rating_diff),
                         self.parameters.double_single_learning_rate),
                Variable(player2_rating, get_setter(self.single_ratings, player2),
                         self.parameters.single_weight / weight_sum * (player2_weight - weight_diff * sig_derivative * scaled_rating_diff),
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


class Parameters(object):
    def __init__(self,
                 learning_rate=30 / Model.SIGMOID_SCALE,
                 double_initial_rating=1500,
                 double_learning_rate=30 / Model.SIGMOID_SCALE,
                 single_weight=100,
                 higher_single_weight=0.9,
                 lower_single_weight=0.5,
                 single_weights_sigmoid_scale=10,
                 double_single_learning_rate=30 / Model.SIGMOID_SCALE):
        self.learning_rate = learning_rate
        self.double_initial_rating = double_initial_rating
        self.double_learning_rate = double_learning_rate
        self.single_weight = single_weight
        self.higher_single_weight = higher_single_weight
        self.lower_single_weight = lower_single_weight
        self.single_weights_sigmoid_scale = single_weights_sigmoid_scale
        self.double_single_learning_rate = double_single_learning_rate

    def validate(self):
        return (self.learning_rate > 0 and
                self.double_initial_rating > 0 and
                self.double_learning_rate > 0 and
                self.single_weight > 0 and
                self.higher_single_weight >= self.lower_single_weight > 0 and
                self.single_weights_sigmoid_scale > 0 and
                self.double_single_learning_rate > 0)

    def to_vector(self):
        return np.array([self.learning_rate, self.double_initial_rating, self.double_learning_rate, self.single_weight,
                         self.higher_single_weight, self.lower_single_weight,
                         self.single_weights_sigmoid_scale,
                         self.double_single_learning_rate])

    @staticmethod
    def from_vector(vector):
        parameters = Parameters(*vector)
        assert np.square(parameters.to_vector() - vector).sum() == 0
        return parameters

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(d):
        return Parameters(**d)


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x)) if x >= 0 else 1.0 - 1.0 / (1 + np.exp(x))


def sigmoid_derivative(x):
    exp = np.exp(-abs(x))
    return exp / (1 + exp) ** 2


def evaluate_model(model, records, start=None, end=None):
    count = 0
    ll_sum = 0.0
    for index, record in enumerate(records):
        if (start is None or index >= start) and (end is None or index < end):
            p = model.predict(record.first_team, record.second_team, record.date)
            if not 0 <= p <= 1:
                model.predict(record.first_team, record.second_team, record.date)
            assert 0 <= p <= 1
            assert record.first_score != record.second_score
            p_corrected = p if record.first_score > record.second_score else 1 - p
            ll_sum += np.log(p_corrected)
            count += 1
        model.update(record)
    return np.exp(ll_sum / count)


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
    model = Model(args.parameters)
    records = list(read(args.input))
    result = evaluate_model(model, records, *args.bound)
    print 'Result:', result


def tune(args):
    records = list(read(args.input))
    initial_vector = args.parameters.to_vector()

    def func(vector):
        parameters = Parameters.from_vector(vector)
        if not parameters.validate():
            return None
        model = Model(parameters)
        return evaluate_model(model, records, *args.bound)

    result, best_result = maximize(func, initial_vector, args.tunelimit)
    best_parameters = Parameters.from_vector(result)
    # best_result = evaluate_model(Model(best_parameters), records, *args.bound)
    print 'Best result:', best_result
    json.dump(best_parameters.to_dict(), args.output)
    args.output.write('\n')
    # for vec in iterate_vectors(initial_vector):
    #     result = optimize.fmin(func, vec)
    #     parameters = Parameters.from_vector(result)
    #     print 'Result:', evaluate_model(Model(parameters), records, *args.bound)
    #     print 'Parameters:', parameters.to_dict()


def maximize(func, initial_vector, limit):
    best_vector = initial_vector
    best_value = func(initial_vector)
    print 'Initial value:', best_value, best_vector
    neg = lambda x: -x if x is not None else None
    func_wrapper = lambda x: neg(func(x))
    for _ in xrange(limit):
        vector = best_vector * np.random.lognormal(0, 1, (len(initial_vector),))
        opt_vector, value, _, _, _ = optimize.fmin(func_wrapper, vector, full_output=True)
        if best_value is None or -value > best_value:
            best_vector = opt_vector
            best_value = -value
            print 'New best value:', best_value, best_vector
    return best_vector, best_value


def iterate_vectors(vector):
    for i in xrange(len(vector)):
        x = vector[i]
        vector[i] = 10 * x
        yield vector
        vector[i] = x / 10
        yield vector
        vector[i] = x


def parse_bounds(s):
    start, end = s.split(':')
    return int(start) if start else None, int(end) if end else None


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
    parser.add_argument('--bound', type=parse_bounds, default=(None, None),
                        help='Bounds for input records, <start>:<end>')
    parser.add_argument('--params', dest='parameters', type=read_parameters, default=Parameters(),
                        help='File with model parameters')
    parser.add_argument('--checkgrad', default=False, action='store_true',
                        help='Check that gradient is correct on every update of the model')
    parser.add_argument('--tunelimit', type=int, default=1,
                        help='Number of times to make leaps from local optima while tuning')
    # parser.add_argument('--reg', dest='regularizer', type=float, default=0.000000003,
    #                     help='Regularizer for tuning parameters')
    args = parser.parse_args()
    if args.checkgrad:
        Model.CHECK_GRADIENT = True
    args.func(args)


if __name__ == '__main__':
    main()
