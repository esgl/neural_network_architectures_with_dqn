import os
import datetime
log_dir = os.path.join("log",
                  datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))