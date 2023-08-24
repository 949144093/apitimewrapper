import builtins
import logging


def print(*args, **kwargs):
    logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                        filename='performance.log',
                        filemode='w',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                        # a是追加模式，默认如果不写的话，就是追加模式
                        format=
                        '%(asctime)s - %(levelname)s: %(message)s'
                        # 日志格式
                        )
    logging.info(*args, **kwargs)
    return builtins.print(*args, **kwargs)