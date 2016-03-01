#!/usr/bin/env python
# coding=utf-8
import argparse
import json
import operator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=argparse.FileType(), help='File with rating data to format')
    parser.add_argument('output', type=argparse.FileType('w'), help='Output file')
    parser.add_argument('--prev', type=argparse.FileType(), help='File with previous rating data')
    parser.add_argument('--names', type=argparse.FileType(), help='File with names')
    args = parser.parse_args()
    format_rating(args.input, args.output, args.prev, args.names)


def format_rating(rating_input, output, prev_rating_input, names_input):
    print >> output, '#|'
    print >> output, '|| **№** | **Игрок** | **Рейтинг** | **Изменение рейтинга** | **Изменение места** ||'
    prev_ratings = {}
    prev_places = {}
    if prev_rating_input is not None:
        for item in json.load(prev_rating_input):
            name = item['name']
            prev_ratings[name] = int(round(item['rating']))
            if 'place' in item:
                prev_places[name] = item['place']
    names = {}
    if names_input is not None:
        for line in names_input:
            alias, name = line.strip().split('\t')
            names[unicode(name, 'utf-8')] = alias
    items = [item for item in json.load(rating_input) if 'place' in item]
    items.sort(key=operator.itemgetter('place'))
    for item in items:
        name = item['name']
        place = item['place']
        rating = int(round(item['rating']))
        rating_diff = None
        place_diff = None
        if name in prev_ratings:
            rating_diff = rating - prev_ratings[name]
        if name in prev_places:
            place_diff = prev_places[name] - place
        print_name = names.get(name, '**' + name + '**')
        print_rating = '**%d**' % rating
        print_rating_diff = format_delta(rating_diff)
        print_place_diff = format_delta(place_diff)
        print >> output, ('|| ' +
                          ' | '.join([str(place), print_name, print_rating, print_rating_diff, print_place_diff]).encode('utf-8') +
                          ' ||')
    print >> output, '|#'


def format_delta(delta):
    if delta is None:
        return '!!(green)NEW!!'
    elif delta > 0:
        return '!!(green)%+d!!' % delta
    elif delta < 0:
        return '!!(red)%d!!' % delta
    else:
        return '!!(grey)0!!'


if __name__ == '__main__':
    main()
