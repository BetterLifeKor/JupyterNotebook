# Day_02_05_matplotlib_balance.py
import xlrd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import font_manager, rc, colors

path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname=path).get_name()
rc('font', family=font_name)


def read_import_export():
    wb = xlrd.open_workbook('Data/국가별수출입 실적_201711305.xls')
    print(wb)

    sheets = wb.sheets()
    print(sheets)

    sheet = sheets[0]
    print(sheet.nrows)

    result = []
    for row in range(6, sheet.nrows):
        # print(sheet.row_values(row))

        values = sheet.row_values(row)

        country = values[1]
        outcome = int(values[3].replace(',', ''))
        income  = int(values[5].replace(',', ''))
        balance = int(values[6].replace(',', ''))

        result.append([country, outcome, income, balance])

    return result


def sorted_top10(result):
    result_sorted = sorted(result,
                           key=lambda t: t[-1],
                           reverse=True)
    # print(*result_sorted, sep='\n')

    red_names, red_balances = [], []
    # for row in result_sorted[:10]:
    #     print(row)

    for country, _, _, balance in result_sorted[:-11:-1]:
        # print(country, balance)
        red_names.append(country)
        red_balances.append(balance)

    black_names, black_balances = [], []
    for country, _, _, balance in result_sorted[:10]:
        black_names.append(country)
        black_balances.append(balance)

    #      적자 10개,  적자 금액      흑자 10개,    흑자 금액
    return red_names, red_balances, black_names, black_balances


def draw_balance(names, balances):
    formatter = FuncFormatter(lambda x, pos: int(x // 10000))
    _, ax = plt.subplots()
    ax.yaxis.set_major_formatter(formatter)

    plt.bar(range(len(names)), balances, color=colors.TABLEAU_COLORS)
    plt.xticks(range(len(names)), names, rotation='vertical')
    plt.show()


def draw_balance_together(red_names, red_balances, black_names, black_balances):
    formatter = FuncFormatter(lambda x, pos: int(x // 10000))
    _, ax = plt.subplots()
    ax.yaxis.set_major_formatter(formatter)

    names = black_names + red_names[::-1]
    balances = black_balances + red_balances[::-1]

    plt.bar(range(len(names)), balances,
            color=['black'] * 10 + ['red'] * 10)
    plt.xticks(range(len(names)), names, rotation='vertical')
    plt.show()


# 문제
# 1. 흑자 상위 10개국에 대해 막대 그래프를 그리세요.
# 2. 적자 상위 10개국에 대해 막대 그래프를 그리세요.
# 3. 흑자/적자를 하나의 그래프에 표현해 주세요. (왼쪽 10개, 오른쪽 10개)

result = read_import_export()
# print(*result, sep='\n')

red_names, red_balances, black_names, black_balances = sorted_top10(result)
# draw_balance(black_names, black_balances)
# draw_balance(red_names, red_balances)
draw_balance_together(red_names, red_balances, black_names, black_balances)

# a = [1, 5, 9]
# print(a)
# print(*a)
# print(1, 5, 9)


def not_used():
    print('red' * 3)
    print(['red'] * 3)
    print([['red'] * 3, ['black'] *3])
    print(['red'] * 3, ['black'] *3)
    print(['red'] * 3 + ['black'] *3)

    # redredred
    # ['red', 'red', 'red']
    # [['red', 'red', 'red'], ['black', 'black', 'black']]
    # ['red', 'red', 'red'] ['black', 'black', 'black']
    # ['red', 'red', 'red', 'black', 'black', 'black']
