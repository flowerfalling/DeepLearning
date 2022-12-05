# -*- coding: utf-8 -*-
# @Time    : 2022/10/4 21:52
# @Author  : 之落花--falling_flowers
# @File    : demo.py
# @Software: PyCharm

def main():
    a = [i for i in range(10)]
    for i in a:
        a.remove(i)
        print(i)
    pass


if __name__ == '__main__':
    main()
    pass
