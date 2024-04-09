import os

class DataFile:
    def __init__(self):
        self.filename = None
        self.page = None
        self.order = None
        self.type = ''
        self.period = []
        self.parts = []
        pass

    def archive(self, file):
        self.filename = file
        items = file.split('-')
        self.page = int(items[0])
        self.order = int(items[1])
        self.type = items[2]
        self.period = items[3].split('_')
        self.parts = items[4].split('+')

    def __str__(self):
        return f'page: {self.page}, order: {self.order}, type: {self.type}, period: {self.period}, parts: {self.parts}'

    def to_name(self) -> str:
        period_str = ''
        for p in self.period:
            period_str += f'_{p}'
        period_str = period_str[1:]
        parts_str = ''
        for p in self.parts:
            parts_str += f'+{p}'
        parts_str = parts_str[1:]
        return f'{self.page}-{self.order}-{self.type}-{period_str}-{parts_str}'

class Dataset:
    def __init__(self, dataset_path='./'):
        self.dataset_path = dataset_path
        self.data_files = []
        self.periods = []
        self.parts = ['head', 'body']

    def load(self):
        periods = set()
        for file in os.listdir(self.dataset_path):
            if file.endswith('.png'):
                file_name = file.split('.')[0]
                data = DataFile()
                #print(file_name)
                data.archive(file_name)
                self.data_files.append(data)
                periods.update(data.period)
        self.periods = list(periods)
        self.periods.sort()

    def file_paths(self):
        return [os.path.join(self.dataset_path, f'{d.filename}.png') for d in self.data_files]
    
    def file_paths_if(self, select):
        return [os.path.join(self.dataset_path, f'{d.filename}.png') for d in self.data_files if select(d)]

def LoadAllFiles(dataset_path='./longxing_dataset_english'):
    dataset = Dataset(dataset_path)
    dataset.load()
    return dataset

if __name__ == '__main__':
    dataset = LoadAllFiles()
    print(dataset.file_paths())