import os
import time
import logging
from datetime import date

this_folder = os.path.dirname(os.path.abspath(__file__))
now = time.time()

try:
    # Deleting log files older than 7 days
    for file in os.listdir(os.path.join(this_folder, 'logs')):
        file_path = os.path.join(this_folder, 'logs\\' + file)

        if os.stat(file_path).st_mtime < (now - 7 * 86400):
            os.remove(file_path)

except Exception as x:
    print('Error in handling log files : ', str(x))

file_date = date.today().strftime('%d-%b-%Y')
log_file_name = os.path.join(this_folder, "logs\\Log_{0}.txt".format(file_date))
logging.basicConfig(filemode='a+',
                    datefmt='%H:%M:%S',
                    level=logging.INFO,
                    filename=log_file_name,
                    format='%(asctime)s - %(levelname)s: %(message)s')
