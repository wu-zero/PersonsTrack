import logging
from logging import handlers
import multiprocessing_logging


# DEBUG	    详细信息，一般只在调试问题时使用。
# INFO	    证明事情按预期工作。
# WARNING	某些没有预料到的事件的提示，或者在将来可能会出现的问题提示。例如：磁盘空间不足。但是软件还是会照常运行。
# ERROR	    由于更严重的问题，软件已不能执行一些功能了。
# CRITICAL


# 2019-11-30 15:50:44,032 - /home/wyw/Documents/PersonsTrack2/persons_track/Logger.py[line:40]
fmt1 = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
fmt2 = '%(levelname)s: %(message)s'


class Logger:
    def __init__(self, filename, file_level='ERROR', terminal_flag=True, terminal_level='INFO'):
        self.logger = logging.getLogger(filename)
        self.logger.setLevel('INFO')  # 设置日志级别
        self.logger.handlers.clear()
        format_str_for_log_file = logging.Formatter(fmt1)
        format_str_for_terminal = logging.Formatter(fmt2)  # 设置日志格式


        # 文件===========================================================================================================
        # 指定间隔时间自动生成文件的处理器
        log_file_handler = handlers.TimedRotatingFileHandler(filename=filename, when='M', backupCount=3, encoding='utf-8')
        # 实例化TimedRotatingFileHandler
        # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位
        log_file_handler.setFormatter(format_str_for_log_file)#设置文件里写入的格式
        log_file_handler.setLevel(file_level)
        # 添加handler到logger============================================================================================
        self.logger.addHandler(log_file_handler)


        # 终端===========================================================================================================
        if terminal_flag:
            terminal_handler = logging.StreamHandler()  # 往屏幕上输出
            terminal_handler.setFormatter(format_str_for_terminal)  # 设置屏幕上显示的格式
            terminal_handler.setLevel(terminal_level)
            self.logger.addHandler(terminal_handler)


        multiprocessing_logging.install_mp_handler()



if __name__ == '__main__':
    log = Logger('all.log')
    log.logger.debug('debug')
    log.logger.info('info')
    log.logger.warning('警告')
    log.logger.error('报错')
    log.logger.critical('严重')
