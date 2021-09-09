from datetime import datetime, timezone, timedelta


class ID_generator():
    def __init__(self):
        self.now = datetime.now(timezone(timedelta(hours=9)))

    def time_stamp(self):
        now = datetime.now(timezone(timedelta(hours=9)))
        return now.strftime('%H%M%S')

    def date_stamp(self):
        now = datetime.now(timezone(timedelta(hours=9)))
        return now.strftime('%Y%m%d')

    def now_stamp(self):
        now = datetime.now(timezone(timedelta(hours=9)))
        return now.strftime('%Y%m%d%H%M%S')
