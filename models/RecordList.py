from models.Record import Record


class RecordList:
    def __init__(self, data, num=None):
        self.data = data
        self.num = num

    def __str__(self):
        print('Shape: ', self.data.shape, ' Num: ', self.num, '\n')

    def toList(self):
        limit = max(0, min(self.num, self.data.shape[0]) if self.num else self.data.shape[0])
        records = [Record(self.data.columns, self.data.iloc[i].values).toDictionary() for i in range(limit)]
        return records
