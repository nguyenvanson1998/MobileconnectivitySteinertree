import logging

# Tạo logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# Tạo file handler và thiết lập mức độ ghi log
file_handler = logging.FileHandler('log_file.log')
file_handler.setLevel(logging.DEBUG)

# Tạo formatter để định dạng thông điệp ghi log
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Thiết lập formatter cho file handler
file_handler.setFormatter(formatter)

# Thêm file handler vào logger
logger.addHandler(file_handler)

# Ghi log
logger.debug('Đây là một thông điệp DEBUG')
logger.info('Đây là một thông điệp INFO')
logger.warning('Đây là một thông điệp WARNING')
logger.error('Đây là một thông điệp ERROR')
logger.critical('Đây là một thông điệp CRITICAL')