#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
import re
import time
import calendar
from datetime import datetime, timedelta, timezone
import json
import os
import pandas as pd
import numpy as np
import sys
from inspect import isfunction
from dateutil.parser import parse
import prettytable as pt
from pandas._libs.tslibs.timestamps import Timestamp


def interval_to_milliseconds(interval):
    """Convert a interval string to milliseconds

    :param interval: Binance interval string, e.g.: 60s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w
    :type interval: str

    :return:
         int value of interval in milliseconds
         None if interval prefix is not a decimal integer
         None if interval suffix is not one of m, h, d, w

    """
    if type(interval) == int:
        return interval
    seconds_per_unit = {
        "s": 1,
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60,
    }
    try:
        return int(interval[:-1]) * seconds_per_unit[interval[-1]] * 1000
    except (ValueError, KeyError):
        return None


def get_month_start_end(date):
    _d = to_datetime(date)
    start = datetime(year=_d.year, month=_d.month, day=1)
    days_num = calendar.monthrange(start.year, start.month)[1]
    end = start + timedelta(days_num)
    return [start, end]


def get_month_range(start, end):
    start = to_datetime(start)
    end = to_datetime(end)
    months = (end.year - start.year) * 12 + end.month - start.month
    month_range = ['%4d-%02d-01' % (start.year + mon // 12, mon % 12 + 1) for mon in range(start.month - 1, start.month + months)]
    return [to_datetime(x) for x in month_range]


def get_year_range(start, end):
    start = to_datetime(start)
    end = to_datetime(end)
    years = end.year - start.year + 2
    year_range = ['%4d/01/01' % (year) for year in range(start.year, start.year + years)]
    return [to_datetime(x) for x in year_range]


def get_cycle_datetime_areas(start_time, end_time, interval):
    # https://blog.csdn.net/qq_36387683/article/details/84766610
    ret = []
    cycle_start_time = start_time
    cycle_ms = interval_to_milliseconds(interval)
    while cycle_start_time < end_time:
        cycle_end_time = cycle_start_time + timedelta(milliseconds=cycle_ms)
        ret.append({'start_time': cycle_start_time, 'end_time': cycle_end_time})
        cycle_start_time = cycle_end_time
    return ret


def run_cycle_datetime_areas(start_time, end_time, interval, func):
    cycle_start_time = start_time
    cycle_ms = interval_to_milliseconds(interval)
    while cycle_start_time < end_time:
        cycle_end_time = cycle_start_time + timedelta(milliseconds=cycle_ms)
        func(start_time=cycle_start_time, end_time=cycle_end_time)
        cycle_start_time = cycle_end_time


def format_date(param, format='%y/%m/%d'):
    return format_datetime(param, format=format)


def format_datetime(param, format='%y/%m/%d %H:%M:%S'):
    if param is None:
        return ''
    # return datetime.fromtimestamp(value / 1000).strftime('%Y-%m-%d %H:%M:%S.%f')
    _d = to_datetime(param)
    if _d is not None:
        return _d.strftime(format)
    return ''


def clear_screen():
    sys.stdout.write('\033[2J')


def clear_line():
    sys.stdout.write('\033[2K')


def to_time(param, fmt=None):
    if param is None:
        return None
    _time = param
    if type(param) is datetime:
        _time = param.time()
    if type(param) is str:
        if fmt is None:
            _time = parse(param).time()
        else:
            _time = datetime.strptime(param, fmt).time()
    return _time


def to_datetime(param, fmt=None):
    if param is None:
        return param
    if type(param) is datetime:
        return param
    if type(param) is pd.Timestamp:
        return pd.to_datetime(param)
    if type(param) is str:
        if fmt is not None:
            return datetime.strptime(param, fmt)
        if param.isdigit():
            param = int(param)
        elif is_float(param):
            param = float(param)
        else:
            return parse(param)
    if type(param) in [int, float]:
        if param > 10000000000:
            param = param / 1000
        return datetime.fromtimestamp(param)
    return datetime.fromtimestamp(param / 1000)


# 最大公约数
def gcd(x, y):
    m = max(x, y)
    n = min(x, y)
    while m % n:
        m, n = n, m % n
    return n


# 最小公倍数
def get_lcm(x, y):
    m = max(x, y)
    n = min(x, y)
    while m % n:
        m, n = n, m % n
    return x * y // n


def round_datetime(param, interval):
    if interval.endswith('d'):
        d = to_datetime(param)
        # 涉及到时区，必须特殊处理
        days = int(interval[:-1])
        days = d.day % days
        td = timedelta(days=days, seconds=d.second, microseconds=d.microsecond, milliseconds=0, minutes=d.minute, hours=d.hour, weeks=0)
        return d - td
    elif interval.endswith('month'):  # 月
        d = to_datetime(param)
        return datetime(year=d.year, month=d.month, day=1)
    elif interval.endswith('y'):
        d = to_datetime(param)
        years = int(interval[:-1])
        year = d.year - d.year % years
        return datetime(year=year, month=1, day=1)
    else:
        d = to_timestamp(param)
        r = interval_to_milliseconds(interval)
        d = d - d % r
        return to_datetime(d)


def is_in_time(param, start, end):
    _time = None
    _start = to_time(start)
    _end = to_time(end)
    if type(param) is datetime:
        _time = param.time()
    else:
        _time = to_time(param)
    # print(param, start, end, _time, _start, _end)
    if _start is not None and _end is not None:
        if _start == _end == _time:
            return True
        if _start < _end:
            return _start <= _time < _end
        else:
            return not _end < _time <= _start
    if _start is None:
        return _time < _end
    if _end is None:
        return _time >= _start


def is_in_docker():
    return os.path.exists("/.dockerenv")


def get_names_by_symbol(name):
    if name.isalpha():
        if name[-3:].upper() in ['ETH', 'BTC', 'BNB']:
            return name[:-3], name[-3:]
        if name[-4:].upper() in ['USDT']:
            return name[:-4], name[-4:]
    return "", ""


def get_datetime_now():
    return datetime.now()


def get_day_begin(day=None):
    if day is None:
        day = datetime.today()
    return datetime(day.year, day.month, day.day, 0, 0, 0)


def to_timestamp(param, fmt=None):
    if param is None:
        return param
    if type(param) in [int, float] and param > 10000000000:
        return param
    if type(param) is timedelta:
        return param.total_seconds() * 1000
    return int(to_datetime(param, fmt=fmt).timestamp() * 1000)


def to_time_seconds(param, fmt=None):
    if param is None:
        return param
    return int(to_datetime(param, fmt=fmt).timestamp())


def to_human_seconds(param):
    if type(param) is timedelta:
        param = param.total_seconds()
    if param < 60:
        return str(int(param)) + '秒'
    if param < 60 * 60:
        return str(param // 60) + '分钟'
    if param < 60 * 60 * 24 * 2:
        return str(param // 3600) + '小时'
    else:
        return str(param // (3600 * 24)) + '天'


def utc2local(utc_st):
    """UTC时间转本地时间（+8: 00）"""
    now_stamp = time.time()
    local_time = datetime.fromtimestamp(now_stamp)
    utc_time = datetime.utcfromtimestamp(now_stamp)
    offset = local_time - utc_time
    local_st = utc_st + offset
    return local_st


def local2utc(local_st):
    """本地时间转UTC时间（-8: 00）"""
    utc_st = datetime.utcfromtimestamp(local_st.timestamp())
    return utc_st


def to_iso8601(param):
    timestamp = to_timestamp(param)
    if timestamp is None:
        return timestamp
    if not isinstance(timestamp, int):
        return None
    if int(timestamp) < 0:
        return None

    try:
        utc = datetime.utcfromtimestamp(timestamp // 1000)
        return utc.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-6] + "{:03d}".format(int(timestamp) % 1000) + 'Z'
    except (TypeError, OverflowError, OSError):
        return None


def parse8601(timestamp=None):
    if timestamp is None:
        return timestamp
    yyyy = '([0-9]{4})-?'
    mm = '([0-9]{2})-?'
    dd = '([0-9]{2})(?:T|[\\s])?'
    h = '([0-9]{2}):?'
    m = '([0-9]{2}):?'
    s = '([0-9]{2})'
    ms = '(\\.[0-9]{1,3})?'
    tz = '(?:(\\+|\\-)([0-9]{2})\\:?([0-9]{2})|Z)?'
    regex = r'' + yyyy + mm + dd + h + m + s + ms + tz
    try:
        match = re.search(regex, timestamp, re.IGNORECASE)
        if match is None:
            return None
        yyyy, mm, dd, h, m, s, ms, sign, hours, minutes = match.groups()
        ms = ms or '.000'
        msint = int(ms[1:])
        sign = sign or ''
        sign = int(sign + '1') * -1
        hours = int(hours or 0) * sign
        minutes = int(minutes or 0) * sign
        offset = timedelta(hours=hours, minutes=minutes)
        string = yyyy + mm + dd + h + m + s + ms + 'Z'
        dt = datetime.strptime(string, "%Y%m%d%H%M%S.%fZ")
        dt = dt + offset
        timestamp = calendar.timegm(dt.utctimetuple()) * 1000 + msint
        return to_datetime(timestamp)
    except (TypeError, OverflowError, OSError, ValueError):
        return None


def show_msg(obj, prefix=""):
    json = format_json(obj)
    print(prefix, json)


def is_float(value):
    if type(value) is int:
        return False
    if type(value) is float:
        return True
    if type(value) is not str:
        return False
    if value.isdigit():
        return False
    try:
        float(value)
        return not value.isnumeric()
    except ValueError:
        return False


def to_bool(value):
    if str(value).lower() in ("yes", "y", "true", "t", "1"):
        return True
    if str(value).lower() in ("no", "n", "false", "f", "0", "0.0", "", "none", "[]", "{}"):
        return False
    raise Exception('Invalid value for boolean conversion: ' + str(value))


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def format_json(obj, color=False, sort_keys=False):
    import json
    if isinstance(obj, str):
        obj = json.loads(obj)
    formatted_json = json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=sort_keys, cls=NpEncoder)
    if color:
        from pygments import highlight, lexers, formatters
        formatted_json = highlight(
            formatted_json.encode('UTF-8'), lexers.JsonLexer(),
            formatters.TerminalFormatter())
    return formatted_json


def rgb_to_html_color(r, g, b):
    """ convert an R, G, B tuple to #RRGGBB """
    hex_color = '#%02x%02x%02x' % (r, g, b)
    # that's it! '%02x' means zero-padded, 2-digit hex values
    return hex_color


def get_pkl_files(dir_name):
    pkl_files = []
    if not os.path.isdir(dir_name):
        print("%s is not a directory." % dir_name)
        return pkl_files
    dir_name = os.path.realpath(dir_name)
    for (root, dirs, files) in os.walk(dir_name):
        for item in files:
            d = os.path.join(root, item)
            if not item.endswith('.pkl'):
                continue
            pkl_files.append(d[len(dir_name) + 1:])

    # files = sorted(os.listdir(dir_name))
    #  files.sort(key=lambda x: os.path.getmtime(x))
    pkl_files.sort()
    pkl_files.reverse()
    return pkl_files


class PrettyTable(pt.PrettyTable):
    def __init__(self, fields={}, data=None, map_func=None, func_params=None, rename={}, hide_cols=[], sum_cols=[], show_index=False,
                 default_float_format='.2', default_align='c', start=0, end=None, title=None, **kwargs):
        super().__init__()
        self._title = title
        if data is None:
            data = {}
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, str):
            data = json.loads(data)
        if isinstance(fields, list):
            fields = dict(zip(fields, fields))
        if isinstance(data, pd.DataFrame):
            columns = data.columns.tolist()
            data = data.to_dict(orient='records')
            if len(fields.keys()) == 0:
                fields = dict(zip(columns, columns))
        if len(fields) == 0 and len(data) == 0:
            return
        if len(fields) == 0 and len(data) > 0:
            keys = data[0].keys()
            fields = dict(zip(keys, keys))
        self._data = data[start:end]
        self._map_func = map_func
        self._func_params = func_params
        self._sum_cols = sum_cols
        self._hide_cols = hide_cols
        self._show_index = False
        self._index = 1
        field_align = {}
        field_float_format = {}
        self.field_time_format = {}
        self.field_format = {}
        field_dict = {}
        for key, field in fields.items():
            name = field
            if key in self._hide_cols:
                continue
            if '|' in field:
                array = field.split("|")
                name = array[0]
                values = array[1:]
                for value in values:
                    m = re.match(r'{(.*)}', value)
                    if m is not None:
                        self.field_format[key] = m.groups()[0]
                    elif value in ('l', 'c', 'r'):
                        field_align[name] = value
                    elif '.' in value:
                        field_float_format[name] = value
                    elif '%' in value:
                        self.field_time_format[key] = value

            if not not rename and key in rename.keys():
                field_dict[key] = rename[key]
            else:
                field_dict[key] = name
        for k, v in kwargs.items():
            if k.endswith("_title"):
                name = k[:k.find('_title')]
                field_dict[name] = v
        fields = list(field_dict.values())
        if show_index and 'index' not in field_dict:
            self._show_index = True
            fields.insert(0, '序号')
        # print(fields)
        self.field_names = fields
        self.float_format = default_float_format
        self.align = default_align
        self.align['序号'] = 'r'
        # print(field_align)
        for name in field_align:
            self.align[name] = field_align[name]
        for name in field_float_format:
            self.float_format[name] = field_float_format[name]
        for k, v in kwargs.items():
            if k.endswith("_float_format"):
                name = k[:k.find('_float_format')]
                self.float_format[field_dict[name]] = v
        self._field_dict = field_dict
        # print(self.field_names)
        # print(self._field_dict)
        if self._data is not None:
            self.add(self._data)

    def add(self, data):
        for row in data:
            if row and self._map_func is not None:
                row = self._map_func(row, self._func_params)
            self.add_row(row)

        sum_row = {}
        for col in self._sum_cols:
            all_cols = []
            for d in self._data:
                if col in d and d[col] != '':
                    all_cols.append(d[col])
            sum_row[col] = sum(all_cols)
        if len(sum_row) > 0:
            self.add_row(sum_row, is_sum=True)

    def _get_node_value(self, value, node):
        if value is None:
            return value
        if "." not in node:
            return value.get(node, '')
        nodes = node.split(".")
        ret = value
        for node in nodes:
            ret = ret.get(node, '')
        return ret

    def add_row(self, obj, is_sum=False):
        row = []
        if self._show_index:
            row.append('all' if is_sum else self._index)
            self._index += 1
        for key in self._field_dict:
            v = self._get_node_value(obj, key)
            if key in self.field_format:
                v = ('{:%s}' % (self.field_format[key])).format(v)
            elif isinstance(v, datetime) or isinstance(v, Timestamp):
                _format = '%y/%m/%d %H:%M:%S'
                if key in self.field_time_format:
                    _format = self.field_time_format[key]
                v = format_datetime(v, _format)
            row.append(v)
        # print(row)
        super().add_row(row)

    def print(self):
        # if self._title:
        #     print_head(self._title)
        print(self)


class QuantConfig():
    def __init__(self, config_path='config.json'):
        if config_path.startswith("/"):
            self._config_path = config_path
        else:
            _dir = os.path.dirname(os.path.abspath(__file__))
            self._config_path = os.path.join(_dir, config_path)
        if not os.path.exists(self._config_path):
            self._settings = self.default_settings()
            self._save_json_config(self._config_path, self._settings)
        else:
            self._settings = self._read_json_config(self._config_path)

    def _read_json_config(self, config_path):
        config_file = open(config_path, 'r')
        data = config_file.read()
        config_file.close()
        return json.loads(data)

    def _save_json_config(self, config_path, config):
        config_file = open(config_path, 'w')
        data = json.dumps(config, indent=4, ensure_ascii=False, sort_keys=True)
        config_file.write(data)
        config_file.close()

    def default_settings(self):
        settings = {}
        return settings

    def get_config(self, node, default=None):
        if isinstance(node, list):
            nodes = node
        else:
            nodes = node.split(".")
        ret = self._settings
        try:
            for node in nodes:
                ret = ret[node]
        except KeyError as e:
            return default
        return ret

    def set_config(self, node, value):
        if isinstance(node, list):
            nodes = node
        else:
            nodes = node.split(".")
        ret = self._settings
        for node in nodes[:-1]:
            if node not in ret:
                ret[node] = {}
            ret = ret[node]
        p_node = nodes[-1]
        if value == ret.get(p_node, None):
            return
        ret[p_node] = value
        self._save_json_config(self._config_path, self._settings)

    def save(self):
        self._save_json_config(self._config_path, self._settings)


def parallel_run(func, params, count=0, thread=False, monitor_func=None):
    from multiprocessing import cpu_count
    from multiprocessing import Pool
    if count == 0:
        count = cpu_count()
    if count >= 4:
        count = count * 2
    if thread:
        from multiprocessing.dummy import Pool as ThreadPool
        pool = ThreadPool(count)
    else:
        pool = Pool(count)
    sync = True
    if monitor_func is not None:
        sync = False
    ret = []
    if sync:
        ret = pool.map(func, params)
    else:

        def over_callback(_rets):
            params.clear()
            for _ret in _rets:
                ret.append(_ret)

        pool.map_async(func, params, callback=over_callback)
    import time
    time.sleep(0.5)
    if monitor_func is not None:
        monitor_func(params)
    pool.close()
    pool.join()
    return ret


def pretty_print(obj):
    print(format_json(obj, True))


def print_head(head, max_len=130, show=True):
    max_sep_len = max_len
    info = " " + head + " "
    text_len = get_text_width(info)
    start = int((max_sep_len - text_len) / 2)
    end = max_sep_len - start - text_len
    msg = start * '-' + info + end * '-'
    if show:
        print(msg)
    return msg


def get_text_width(text):
    len = 0
    for c in text:
        if is_chinese(c):
            len += 2
        else:
            len += 1
    return len


def is_chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


class CheckParamErrorException(Exception):
    def __init__(self, err, msg):
        self._err = err
        self._msg = msg

    def __str__(self):
        return '%s: %s' % (self._err, self._msg)


def check_params(params, msg=""):
    if isinstance(params, list):
        for param in params:
            if not param:
                raise CheckParamErrorException("参数错误", msg)
    if not params:
        raise CheckParamErrorException("参数错误", msg)


def check_none_params(params, msg=""):
    if isinstance(params, list):
        for param in params:
            if param is None:
                raise CheckParamErrorException("参数错误", msg)
    if params is None:
        raise CheckParamErrorException("参数错误", msg)


def call_time_limit(s, t, block=False):
    def wrapper(func):
        name = func.__name__  # 给变量name赋值 确定访问的函数
        func_identify = {name: {'calls': [], 'seconds': s, 'times': t, 'block': block}}

        def inner(*args, **kwargs):
            now = get_datetime_now()
            info = func_identify[name]
            seconds = info['seconds']
            times = info['times']
            calls = info['calls']
            del_count = 0
            for call in calls:
                if (now - call).total_seconds() > seconds:
                    del_count += 1
                else:
                    break
            for c in range(del_count):
                calls.pop(0)
            if len(calls) >= times:
                min_wait_seconds = seconds - (now - calls[0]).total_seconds()
                if info['block']:
                    print("频率限制，阻塞 {:.3f} 秒".format(min_wait_seconds))
                    time.sleep(min_wait_seconds)
                else:
                    print("频率限制，请稍后{:.3f}访问".format(min_wait_seconds))
                    return None
            res = func(*args, **kwargs)
            now = get_datetime_now()
            calls.append(now)
            return res

        return inner

    return wrapper


class GlobalData(dict):
    def __init__(self):
        pass

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            return None


class StopWatch:
    def __init__(self, name='', show_all=False):
        self._name = name
        self._start_time = get_datetime_now()
        self._stat = {}
        self._show_all = show_all

    def start(self):
        self._start_time = get_datetime_now()

    def stop(self, sub):
        self._stat[sub] = get_datetime_now()

    def print(self):
        print(self)

    def __str__(self):
        fields = {'sub_name': '名称', 'start_time': '开始时间', 'end_time': '结束时间', 'seconds': '消耗秒'}
        data = []
        last_time = self._start_time
        for sub, _time in self._stat.items():
            row = {
                'sub_name': sub, 'start_time': last_time, 'end_time': _time,
                'seconds': (_time - last_time).total_seconds()
            }
            data.append(row)
            last_time = _time
        if self._show_all:
            now = get_datetime_now()
            last_row = {
                'sub_name': 'all', 'start_time': self._start_time, 'end_time': now,
                'seconds': (_time - self._start_time).total_seconds()
            }
            data.append(last_row)
        table = PrettyTable(fields, data=data, show_index=True)
        return str(table)


def fuzzy_finder(user_input, collection):
    suggestions = []
    pattern = '.*?'.join(user_input)  # Converts 'djm' to 'd.*?j.*?m'
    regex = re.compile(pattern)  # Compiles a regex.
    for item in collection:
        match = regex.search(item)  # Checks if the current item matches the regex.
        if match:
            suggestions.append((len(match.group()), match.start(), item))
    return [x for _, _, x in sorted(suggestions)]


def run_cmd_shell(argv=[], prefix='cmd-> ', words=[], cmd_func=None):
    if len(argv) > 1 and cmd_func is not None:
        cmd_func(argv[1:])
        return
    from prompt_toolkit import prompt, PromptSession
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.lexers import PygmentsLexer
    from prompt_toolkit.styles import Style
    from pygments.lexers.sql import SqlLexer
    from prompt_toolkit.completion import FuzzyCompleter
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    _completer = FuzzyCompleter(WordCompleter(words))
    session = PromptSession(lexer=PygmentsLexer(SqlLexer), completer=_completer, auto_suggest=AutoSuggestFromHistory())
    while True:
        try:
            text = session.prompt(prefix)
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        else:
            args_str = text.strip()
            if args_str == '':
                continue
            args = args_str.split(' ')
            # print(args)
            if len(args) > 0 and cmd_func is not None:
                cmd_func(args)


def run_shell(prefix=None, all_cmd_table={}, globals_params={}, params_options={}, module_name="__main__"):
    def _set_default_global_params(key, value):
        if key not in globals_params:
            globals_params[key] = value

    def _process_global_params(args: list):
        _set_default_global_params('debug', False)
        _set_default_global_params('start', to_datetime('2020-06-01 00:00'))
        _set_default_global_params('end', to_datetime('2020-07-01 00:00'))

        cmd_all = {}
        for arg in args[:]:
            if arg == '-':
                break
            cmd = arg
            if '=' in arg:
                cmd = arg[:arg.index('=') + 1]
            search_list = fuzzy_finder(cmd, all_cmd_table.keys())
            if len(search_list) > 1:
                print("%s is %s" % (arg, ','.join(search_list)))
                raise Exception("arg error")
            if len(search_list) == 1:
                _cmd = search_list[0]
                if _cmd in cmd_all.keys() or _cmd in cmd_all.values():
                    print('more cmd ', arg, cmd_all)
                    raise Exception("arg error")
                cmd_all[arg] = _cmd
                if _cmd == arg:
                    args.remove(arg)

        now: datetime = get_datetime_now()
        for arg in args:
            if arg == '-':
                break
            name, value = ("", "")
            if '=' not in arg:
                name = arg
            else:
                name, value = arg.split('=')
            # print(name, '=', value)
            # search_list = list(filter(lambda s: s.startswith(name), globals_params))
            search_list = fuzzy_finder(arg, globals_params.keys())
            if name not in search_list:
                if len(search_list) > 1:
                    print("param error ", search_list)
                    raise Exception("param error")
                if len(search_list) == 1:
                    name = search_list[0]

            if name in globals_params:
                _type = type(globals_params[name])
                if _type is datetime:
                    globals_params[name] = to_datetime(value)
                elif _type is bool and value == '':
                    globals_params[name] = True
                elif _type is int:
                    globals_params[name] = int(value)
                else:
                    globals_params[name] = value
            else:
                if not value:
                    p_search_all = {}
                    for p_name, p_v_list in params_options.items():
                        _search_list = list(filter(lambda x: arg in x.lower(), p_v_list))
                        if _search_list and len(_search_list) == 1:
                            p_search_all[p_name] = _search_list[0]
                    for _key, _item in p_search_all.items():
                        if globals_params.get(_key):
                            print("%s = %s, %s = %s already override" % (_key, globals_params[_key], _key, _item))
                        print("process params: %s -> %s = %s" % (arg, _key, _item))
                        globals_params[_key] = _item
                else:
                    if value.isdigit():
                        globals_params[name] = int(value)
                    elif is_float(value):
                        globals_params[name] = float(value)
                    else:
                        globals_params[name] = value

            # 匹配可选参数
            if name in params_options.keys():
                search_list = list(filter(lambda x: value in x.lower(), params_options[name]))
                if len(search_list) == 1:
                    globals_params[name] = search_list[0]
                    print(f"参数值匹配 {name}: {value} -> {globals_params[name]} ")

            if name == 'month':
                _d = value
                if _d.isdigit():
                    _d = datetime(year=now.year, month=int(value), day=1)
                start, end = get_month_start_end(_d)
                globals_params['start'] = start
                globals_params['end'] = end
            if name in set(map(str, range(1999, 2030))):
                name, value = 'year', name
            if name == 'year':
                _year = int(value)
                start = datetime(year=_year, month=1, day=1)
                end = datetime(year=_year + 1, month=1, day=1)
                globals_params['start'] = start
                globals_params['end'] = end

        return cmd_all

    def cmd_show_help():
        """显示帮助信息"""
        help_info = []
        for cmd, func in all_cmd_table.items():
            help_info.append({'cmd': cmd, 'func_name': func.__name__, 'desc': func.__doc__})
        PrettyTable(data=help_info, default_align='l').print()

    def cmd_show_vars():
        """显示全局变量信息"""
        var = globals_params.get("var")
        if var:
            print('%s = %s' % (var, globals_params.get(var)))
            globals_params.pop('var')
        else:
            info = []
            for k, v in globals_params.items():
                info.append({"name": k, "value": v})
            info.sort(key=lambda x: x['name'])
            PrettyTable(data=info, default_align='l').print()

    def _run_cmd_func(args):
        cmd_all = _process_global_params(args)
        # print(cmd_all)
        for arg, cmd in cmd_all.items():
            cmd_func = all_cmd_table.get(cmd)
            print("run %s -> %s()" % (arg, cmd_func.__name__))
            cmd_func()

    if module_name in sys.modules:
        module = sys.modules[module_name]
        # print(module.__dict__.keys())
        all_cmd_table.update(dict((k[len("cmd_"):], v) for k, v in module.__dict__.items() if isfunction(v) and k.startswith("cmd_")))
        # print(all_cmd_table)
        if prefix is None:
            file_name = os.path.basename(module.__file__)
            if file_name.endswith('.py'):
                prefix = file_name[:-3] + "-> "
    if 'help' not in all_cmd_table:
        all_cmd_table['help'] = cmd_show_help
    if 'vars' not in all_cmd_table:
        all_cmd_table['var='] = cmd_show_vars
        all_cmd_table['vars'] = cmd_show_vars
    if 'debug' not in globals_params.keys():
        globals_params['debug'] = False
    complete_words = list(all_cmd_table.keys())
    for key, options in params_options.items():
        for item in options:
            complete_words.append("%s=%s" % (key, item))
    for _key, value in globals_params.items():
        _type = type(globals_params[_key])
        if _type is bool:
            complete_words.append("%s=true" % (_key))
            complete_words.append("%s=false" % (_key))
        else:
            complete_words.append("%s=" % (_key))
    run_cmd_shell(sys.argv, prefix=prefix, words=complete_words, cmd_func=_run_cmd_func)
